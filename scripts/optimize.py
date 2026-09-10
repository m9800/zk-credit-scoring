"""Original ZKML logical plans + calibrated estimator; no proof timings in selection."""
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
import msgpack
import numpy as np
from scripts.common import ROOT, config, save_json, sha256
from scripts.zk import write_model, write_inputs, run, reference

ESTIMATOR = ROOT / "vendor/zkml/target/release/zkml-estimate"


def parse_estimate(text):
    def value(pattern, cast):
        matches = re.findall(pattern, text)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one estimator field: {pattern}")
        return cast(matches[0])
    result = {
        "estimated_ns": value(r"Total time cost \(esitmated\): ([\d.eE+\-]+) \(ns\)", float),
        "k": value(r"Optimal k: (\d+)", int),
        "arithmetic_rows": value(r"Total number of rows: (\d+)", int),
        "arithmetic_k": value(r"Arithmetic-row k estimate: (\d+)", int),
    }
    if not math.isfinite(result["estimated_ns"]) or result["estimated_ns"] <= 0:
        raise ValueError("Estimator must return a positive, finite cost")
    return result


def choose(candidates, width=None):
    valid = [c for c in candidates if c.get("valid") and (width is None or c["columns"] == width)]
    if not valid:
        raise ValueError("No valid candidate")
    # Same objective as find_optimal.py, with explicit deterministic tie breaking.
    return min(valid, key=lambda c: (c["estimated_ns"], c["implementation"], c["columns"]))


def logical_models(sf):
    directory = ROOT / "artifacts/optimizer/logical"
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "vendor/zkml/python/create_logical.py"),
        "--model", str(ROOT / "artifacts/model/model.tflite"), "--model_output_dir", str(directory),
        "--config_output_dir", str(directory), "--scale_factor", str(sf), "--k", "12",
        "--num_cols", "10", "--num_randoms", str(config()["num_randoms"])]
    with (directory / "conversion.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    paths = sorted(directory.glob("model_*.msgpack"))
    if len(paths) != 6:
        raise ValueError(f"Expected the six original FullyConnected plans, got {len(paths)}")
    models = []
    for path in paths:
        model = msgpack.unpackb(path.read_bytes())
        assert model["global_sf"] == sf
        model["out_idxes"] = sorted(t["idx"] for t in model["tensors"]) + model["out_idxes"]
        model["commit_before"] = []
        model["commit_after"] = []
        models.append(model)
    assert {m["layers"][0]["implementation"] for m in models} == set(range(6))
    assert all(m["tensors"] == models[0]["tensors"] for m in models)
    return models


def feature_corners(preprocessing=ROOT / "artifacts/model/preprocessing.json"):
    state = json.loads(preprocessing.read_text())
    low = (np.array(state["lower"]) - state["means"]) / state["stds"]
    high = (np.array(state["upper"]) - state["means"]) / state["stds"]
    return np.asarray(list(itertools.product(*zip(low, high))), dtype=np.float32)


def main():
    cfg = config()
    calibration_dir = ROOT / "results/calibration"
    calibration = json.loads((calibration_dir / "metadata.json").read_text())
    if calibration["summary_sha256"] != sha256(calibration_dir / "summary.msgpack"):
        raise ValueError("Calibration checksum mismatch")
    if calibration["rayon_threads"] != cfg["rayon_threads"] or calibration["adapter_sha256"] != sha256(ROOT / "build.rs"):
        raise ValueError("Calibration does not match threads/build adapter; recalibrate")
    quant = json.loads((ROOT / "artifacts/model/quantization.json").read_text())
    sf = quant["scale_factor"]
    start = time.perf_counter()
    models = logical_models(sf)
    corners = feature_corners()
    expected, logits = reference(models[0], corners)
    # Numeric-range floor is independent of row layout and held-out test values.
    k_start = max(8, int(np.ceil(np.log2(max(2 * (int(np.abs(logits).max()) + 10), 2 * sf + 20)))))
    env = {**os.environ, "RAYON_NUM_THREADS": str(cfg["rayon_threads"])}
    candidates = []
    capacity_seconds = estimator_seconds = 0.0
    for cols in range(cfg["optimizer_min_columns"], cfg["optimizer_max_columns"] + 1):
        for model in models:
            implementation = model["layers"][0]["implementation"]
            folder = ROOT / f"artifacts/optimizer/candidates/c{cols}_i{implementation}"
            path = folder / "model.msgpack"
            # Check every corner, not only a zero/dummy input. Capacity and numeric
            # feasibility are established before invoking the original cost formula.
            checks = []
            cap_start = time.perf_counter()
            for k in range(k_start, 13):
                variant = write_model(model, path, columns=cols, k=k, implementation=implementation)
                inputs = write_inputs(variant, corners, folder / "inputs")
                try:
                    actual = run("mock", path, inputs, folder / f"capacity_k{k}", timeout=120)
                except RuntimeError:
                    log = folder / f"capacity_k{k}/mock.log"
                    # Do not silently reinterpret an arbitrary implementation error
                    # as insufficient capacity. Preserve diagnostics and fail closed.
                    message = log.read_text()
                    row_overflow = re.search(r"row=\d+, usable_rows=0\.\.\d+, k=\d+", message)
                    if not row_overflow and not any(s in message for s in ("NotEnoughRowsAvailable", "NotEnoughRows", "ConstraintNotSatisfied", "Lookup", "assertion `left == right` failed")):
                        raise
                    checks.append({"k": k, "valid": False, "log": str(log.relative_to(ROOT))})
                    continue
                if [r["score_integer"] for r in actual] != expected.tolist():
                    raise RuntimeError(f"Numerical mismatch: c={cols}, implementation={implementation}")
                checks.append({"k": k, "valid": True})
                break
            else:
                raise RuntimeError(f"No valid k in calibrated range for {folder}")
            capacity_seconds += time.perf_counter() - cap_start
            first_input = json.loads(inputs.read_text())[0]
            estimate_start = time.perf_counter()
            process = subprocess.run([str(ESTIMATOR), str(path), first_input, "kzg"], cwd=calibration_dir,
                                     env=env, capture_output=True, text=True, timeout=30)
            estimator_seconds += time.perf_counter() - estimate_start
            (folder / "estimate.log").write_text(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(f"Estimator failed: {folder / 'estimate.log'}")
            estimate = parse_estimate(process.stdout)
            if estimate["k"] != k:
                raise RuntimeError("Estimator requested a larger k: revalidate and extend calibration before proceeding")
            candidates.append({"valid": True, "columns": cols, "implementation": implementation, **estimate,
                "capacity_checks": checks, "model_path": str(path.relative_to(ROOT)),
                "model_sha256": sha256(path), "range_corners_checked": len(corners)})
        save_json(ROOT / "artifacts/optimizer/progress.json", {"completed": len(candidates), "last_columns": cols})
        if cols % 10 == 0:
            print(f"Validated and estimated {len(candidates)} candidates (through {cols} columns)", flush=True)
    baseline = choose(candidates, cfg["baseline_columns"])
    optimized = choose(candidates)
    for name, choice in (("credit_fixed_40", baseline), ("credit_optimized", optimized)):
        payload = (ROOT / choice["model_path"]).read_bytes()
        (ROOT / "artifacts/model" / f"{name}.msgpack").write_bytes(payload)
    result = {"method": "Original ZKML logical-plan generation and hardware-calibrated cost estimator",
        "selection_uses_real_proof_timings": False, "candidate_count": len(candidates),
        "column_range_inclusive": [cfg["optimizer_min_columns"], cfg["optimizer_max_columns"]],
        "scale_factor": sf, "num_randoms": cfg["num_randoms"], "rayon_threads": cfg["rayon_threads"],
        "tflite_sha256": sha256(ROOT / "artifacts/model/model.tflite"),
        "preprocessing_sha256": sha256(ROOT / "artifacts/model/preprocessing.json"),
        "calibration_sha256": sha256(calibration_dir / "summary.msgpack"),
        "estimator_binary_sha256": sha256(ESTIMATOR), "adapter_sha256": sha256(ROOT / "build.rs"),
        "optimizer_wall_s": time.perf_counter() - start, "capacity_check_wall_s": capacity_seconds,
        "estimator_calls_wall_s": estimator_seconds, "calibration_wall_s": calibration["calibration_wall_s"],
        "baseline_choice": baseline, "optimized_choice": optimized, "candidates": candidates}
    save_json(ROOT / "results/optimizer.json", result)
    print(json.dumps({"baseline": baseline, "optimized": optimized, "optimizer_wall_s": result["optimizer_wall_s"]}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("logistic", "neural"), default="logistic")
    args = parser.parse_args()
    if args.model == "neural":
        from scripts.neural_optimizer import NeuralOptimizer
        NeuralOptimizer().run()
    else:
        main()
