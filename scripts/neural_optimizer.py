"""Per-FC search using upstream ZKML's calibrated estimator and actual circuits.

Estimate every layout/domain first, then check candidates in ascending cost order.
The first passing candidate is the minimum estimated cost among feasible candidates
in the declared search. No monotonic-cost assumption or measured-proof selection.
Selected circuits are bound to model, inputs, calibration, binaries and source.
"""

from dataclasses import dataclass
import itertools
import json
import os
import re
import subprocess
import time

import msgpack
import numpy as np

from scripts.circuit_plan import CircuitPlan
from scripts.common import ROOT, config, save_json, sha256
from scripts.fixed_point import evaluate
from scripts.optimize import ESTIMATOR, feature_corners, parse_estimate
from scripts.zk import BINARY, convert, run, write_inputs, write_model


@dataclass(frozen=True)
class Layout:
    columns: int
    plan: CircuitPlan
    k: int

    @property
    def key(self):
        return f"c{self.columns}_{self.plan.key}_k{self.k}"


def public_domain_floor(model):
    """One public instance column needs one row per constant value plus score.

    This necessary capacity bound is independent of FC implementation. Blinding
    rows and arithmetic can require still larger k, checked by actual circuits.
    """
    public_values = sum(len(t["data"]) for t in model["tensors"]) + 1
    return max(10, (public_values - 1).bit_length())


def plans(model):
    """Expand FC occurrences, not the set of layer types used by create_logical.py."""
    count = sum(layer["layer_type"] == "FullyConnected" for layer in model["layers"])
    return [CircuitPlan(p) for p in itertools.product(range(6), repeat=count)]


def capacity_failure(message):
    # Only recognize actual capacity/constraint failures; crashes/timeouts and
    # unrelated assertions remain errors, never cheap 'infeasible' candidates.
    return bool(re.search(r"row=\d+, usable_rows=0\.\.\d+, k=\d+", message)) or any(
        text in message
        for text in (
            "NotEnoughRowsAvailable",
            "NotEnoughRows",
            "ConstraintNotSatisfied",
            "Lookup",
        )
    )


def cheapest_verified(candidates, verify, width=None):
    """All estimates are complete before checking feasibility or selecting."""
    ordered = sorted(
        (c for c in candidates if width is None or c["columns"] == width),
        key=lambda c: (
            c["estimated_ns"],
            tuple(c["implementations"]),
            c["columns"],
            c["k"],
        ),
    )
    for candidate in ordered:
        if verify(candidate):
            return candidate
    raise ValueError("No feasible candidate in the declared calibrated search")


class NeuralOptimizer:
    def __init__(self):
        self.cfg = config()
        self.model_dir = ROOT / "artifacts/comparison/neural"
        self.work_dir = self.model_dir / "optimizer"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.env = {**os.environ, "RAYON_NUM_THREADS": str(self.cfg["rayon_threads"])}

        self.check_calibration()
        self.prepare_model()
        self.prepare_samples()
        self.record_identity()

    def check_calibration(self):
        """Require calibration from the current build and thread settings."""
        self.calibration_dir = ROOT / "results/calibration"
        self.calibration = json.loads(
            (self.calibration_dir / "metadata.json").read_text()
        )
        if (
            self.calibration["summary_sha256"]
            != sha256(self.calibration_dir / "summary.msgpack")
            or self.calibration["rayon_threads"] != self.cfg["rayon_threads"]
            or self.calibration["adapter_sha256"] != sha256(ROOT / "build.rs")
        ):
            raise ValueError("Calibration mismatch; run make calibrate on this machine")

        coefficients = msgpack.unpackb(
            (self.calibration_dir / "summary.msgpack").read_bytes()
        )
        for operation in ("fft", "msm", "add", "mul", "permute"):
            for k in range(10, 15):
                value = coefficients[f"kzg_{operation}"][str(k)]
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("Incomplete or invalid calibration")

    def prepare_model(self):
        """Convert the validated weights once for all candidate layouts."""
        quantization = json.loads((self.model_dir / "quantization.json").read_text())
        for key, path in (
            ("tflite_sha256", self.model_dir / "model.tflite"),
            ("validation_sha256", ROOT / "artifacts/comparison/validation.npz"),
            ("config_sha256", ROOT / "configs/experiment.json"),
        ):
            if quantization[key] != sha256(path):
                raise ValueError("Quantization inputs changed; rerun validate-neural")

        # Convert once; every layout uses these same quantized weights.
        self.sf = quantization["scale_factor"]
        self.model = convert(
            self.model_dir / "model.tflite", self.work_dir / "converted", self.sf
        )
        if (
            sum(
                layer["layer_type"] == "FullyConnected"
                for layer in self.model["layers"]
            )
            != 3
        ):
            raise ValueError("Expected the three-FC neural candidate")
        self.model_plans = plans(self.model)
        self.minimum_k = public_domain_floor(self.model)
        if self.minimum_k > 12:
            raise ValueError("Public model values exceed the calibrated base domains")

    def prepare_samples(self):
        """Include input corners and validation rows with challenging ranges."""
        validation = np.load(ROOT / "artifacts/comparison/validation.npz")
        x = validation["x"]
        inference = evaluate(self.model, x)
        # Include validation extrema and seeded observations as well as all input
        # corners. ReLU networks can have interior extrema: corners alone do not
        # establish global intermediate bounds. This remains sampled feasibility.
        indices = np.unique(
            np.r_[
                np.argmin(x, axis=0),
                np.argmax(x, axis=0),
                np.argmin(inference.logits),
                np.argmax(inference.logits),
                np.argmax(inference.max_abs_by_row),
                np.argsort(np.abs(inference.logits))[:4],
                np.random.default_rng(self.cfg["seed"]).choice(
                    len(x), min(16, len(x)), replace=False
                ),
            ]
        )
        self.samples = np.concatenate(
            [
                feature_corners(ROOT / "artifacts/comparison/preprocessing.json"),
                x[indices],
            ]
        )
        self.expected = evaluate(self.model, self.samples).scores.tolist()

    def record_identity(self):
        """Bind the search to its inputs so benchmarks reject stale artifacts."""
        inputs = [
            self.model_dir / "model.tflite",
            self.model_dir / "quantization.json",
            self.work_dir / "converted/base.msgpack",
            ROOT / "vendor/zkml/python/converter.py",
            ROOT / "uv.lock",
            ROOT / "artifacts/comparison/preprocessing.json",
            ROOT / "artifacts/comparison/validation.npz",
            ROOT / "artifacts/comparison/comparison.json",
            self.calibration_dir / "summary.msgpack",
            self.calibration_dir / "metadata.json",
            BINARY,
            ESTIMATOR,
            ROOT / "build.rs",
        ]
        source_files = (
            "neural_optimizer.py",
            "optimize.py",
            "zk.py",
            "fixed_point.py",
            "circuit_plan.py",
            "common.py",
        )
        inputs.extend(ROOT / "scripts" / name for name in source_files)
        fingerprints = [[str(path.relative_to(ROOT)), sha256(path)] for path in inputs]
        self.identity = ["neural-search-v1", self.cfg, self.sf, fingerprints]

    def estimate(self, layout):
        # One scratch circuit keeps search disk usage bounded. Selected variants
        # are materialized later from the same frozen converted model.
        folder = self.work_dir / "scratch"
        path = folder / "model.msgpack"
        variant = write_model(
            self.model,
            path,
            columns=layout.columns,
            k=layout.k,
            implementation=layout.plan,
        )
        inputs = write_inputs(variant, self.samples[:1], folder / "inputs")
        first = json.loads(inputs.read_text())[0]
        process = subprocess.run(
            [str(ESTIMATOR), str(path), first, "kzg"],
            cwd=self.calibration_dir,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        row = {
            "key": layout.key,
            "columns": layout.columns,
            "implementations": list(layout.plan.implementations),
            "requested_k": layout.k,
        }

        if process.returncode:
            # Degree extension can exceed k=14 even with a small base domain.
            # This is explicitly outside calibration, not a feasibility success.
            if not re.search(
                r"Missing calibration for kzg_\w+ at k=\d+", process.stderr
            ):
                (folder / "estimate-error.log").write_text(
                    process.stdout + process.stderr
                )
                raise RuntimeError(f"Estimator failed: {folder / 'estimate-error.log'}")
            row["status"] = "outside_calibration"
        else:
            row.update(parse_estimate(process.stdout))
            row["status"] = "estimated" if row["k"] <= 12 else "outside_calibration"

        return row

    def verify(self, candidate):
        layout = Layout(
            candidate["columns"],
            CircuitPlan(tuple(candidate["implementations"])),
            candidate["k"],
        )
        folder = self.work_dir / "checked" / layout.key
        path = folder / "model.msgpack"
        variant = write_model(
            self.model,
            path,
            columns=layout.columns,
            k=layout.k,
            implementation=layout.plan,
        )
        inputs = write_inputs(variant, self.samples, folder / "inputs")

        try:
            actual = run("mock", path, inputs, folder / "mock", timeout=600)
        except RuntimeError:
            if not capacity_failure((folder / "mock/mock.log").read_text()):
                raise
            return False

        if [row["score_integer"] for row in actual] != self.expected:
            raise RuntimeError(f"Circuit/reference mismatch: {layout.key}")

        candidate.update(
            {
                "valid": True,
                "model_path": str(path.relative_to(ROOT)),
                "model_sha256": sha256(path),
                "mock_observations": len(self.samples),
            }
        )
        return True

    def run(self):
        start = time.perf_counter()
        rows = {}
        outside = 0
        min_columns = self.cfg["optimizer_min_columns"]
        max_columns = self.cfg["optimizer_max_columns"]

        for columns in range(min_columns, max_columns + 1):
            for plan in self.model_plans:
                # k=10..12, extended FFT/arithmetic domains through k=14.
                # Enumerate domains rather than assume calibration is monotonic.
                for k in range(self.minimum_k, 13):
                    row = self.estimate(Layout(columns, plan, k))
                    if row["status"] == "outside_calibration":
                        outside += 1
                    else:
                        rows[(columns, plan, row["k"])] = row
            print(
                f"Estimated through {columns} columns: {len(rows)} distinct calibrated candidates",
                flush=True,
            )

        # Select by estimated cost, validating feasibility before accepting.
        candidates = list(rows.values())
        fixed = cheapest_verified(candidates, self.verify, self.cfg["baseline_columns"])
        optimized = cheapest_verified(candidates, self.verify)
        for name, choice in (
            ("credit_fixed_40", fixed),
            ("credit_optimized", optimized),
        ):
            (self.model_dir / f"{name}.msgpack").write_bytes(
                (ROOT / choice["model_path"]).read_bytes()
            )

        report = {
            "method": "Per-FC Cartesian plans with original ZKML calibrated cost estimator",
            "selection_uses_real_proof_timings": False,
            "candidate_count": len(candidates),
            "logical_plan_count": len(self.model_plans),
            "outside_calibration_count": outside,
            "candidates": candidates,
            "column_range_inclusive": [min_columns, max_columns],
            "k_range_inclusive": [self.minimum_k, 12],
            "public_capacity_pruned_domains": (self.minimum_k - 10)
            * len(self.model_plans)
            * (max_columns - min_columns + 1),
            "feasibility": "Sampled validation observations and all 16 feature corners; not a global range proof",
            "sampled_max_abs_intermediate": evaluate(
                self.model, self.samples
            ).max_abs_intermediate,
            "scale_factor": self.sf,
            "rayon_threads": self.cfg["rayon_threads"],
            "baseline_choice": fixed,
            "optimized_choice": optimized,
            "invocation_wall_s": time.perf_counter() - start,
            "calibration_wall_s": self.calibration["calibration_wall_s"],
            "identity": self.identity,
        }
        save_json(self.model_dir / "optimizer.json", report)
        print(
            json.dumps({"fixed_40": fixed, "optimized": optimized}, indent=2),
            flush=True,
        )
