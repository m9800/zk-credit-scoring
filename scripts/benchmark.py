"""Bounded empirical configuration tuning, explicitly not the paper's cost estimator."""
import argparse
import itertools
import json
import platform
import subprocess
import time
import numpy as np
from scripts.common import ROOT, config, save_json, sha256
from scripts.zk import convert, write_model, write_inputs, run, reference, BINARY


def summary(report):
    p = np.array([m["proving_s"] for m in report["measurements"]])
    v = np.array([m["verification_s"] for m in report["measurements"]])
    return {"median_proving_s": float(np.median(p)), "proving_iqr_s": float(np.quantile(p, .75) - np.quantile(p, .25)),
            "median_verification_s": float(np.median(v)), "proof_bytes": report["measurements"][0]["proof_bytes"],
            "peak_process_rss_bytes": report["peak_process_rss_bytes"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, nargs="+")
    parser.add_argument("--tune-only", action="store_true")
    args = parser.parse_args()
    cfg = config()
    columns = args.columns or cfg["search_columns"]
    if cfg["baseline_columns"] not in columns:
        raise ValueError("The baseline width must be in the search space")
    quantization = json.loads((ROOT / "artifacts/model/quantization.json").read_text())
    sf = quantization["scale_factor"]
    model = convert(ROOT / "artifacts/model/model.tflite", ROOT / "artifacts/zk/search", sf)
    state = json.loads((ROOT / "artifacts/model/preprocessing.json").read_text())
    low = (np.array(state["lower"]) - state["means"]) / state["stds"]
    high = (np.array(state["upper"]) - state["means"]) / state["stds"]
    corners = np.asarray(list(itertools.product(*zip(low, high))), dtype=np.float32)
    _, logits = reference(model, corners)
    # All possible preprocessed observations fit these fixed-point ranges.
    k_start = max(8, int(np.ceil(np.log2(max(2 * (int(np.abs(logits).max()) + 10), 2 * sf + 20)))))
    val = np.load(ROOT / "data/processed/validation.npz")
    tune_indices = np.random.default_rng(cfg["seed"]).choice(len(val["x"]), 3, replace=False)
    tune_x = val["x"][tune_indices]
    search_manifest = {"model_sha256": sha256(ROOT / "artifacts/model/model.tflite"),
                       "scale": sf, "columns": columns, "implementations": list(range(6)),
                       "validation_rows": val["row_id"][tune_indices].tolist(), "k_start": k_start,
                       "rayon_threads": cfg["rayon_threads"]}
    search_start = time.perf_counter()
    candidates = []
    for cols, implementation in itertools.product(columns, range(6)):
        folder = ROOT / f"artifacts/runs/tune_c{cols}_i{implementation}"
        path = folder / "model.msgpack"
        capacity_errors = []
        for k in range(k_start, 17):
            variant = write_model(model, path, columns=cols, k=k, implementation=implementation)
            check_inputs = write_inputs(variant, corners, folder / "range_inputs")
            try:
                actual = run("mock", path, check_inputs, folder / "capacity", timeout=120)
            except RuntimeError as error:
                capacity_errors.append({"k": k, "error": str(error)})
                continue
            if [r["score_integer"] for r in actual] != reference(variant, corners)[0].tolist():
                raise RuntimeError(f"Implementation {implementation} changes the declared fixed-point output")
            break
        else:
            raise RuntimeError(f"No valid bounded k for c={cols}, i={implementation}")
        inputs = write_inputs(variant, tune_x, folder / "inputs")
        report = run("prove", path, inputs, folder / "proof", timeout=300)
        if [r["score_integer"] for r in report["measurements"]] != reference(variant, tune_x)[0].tolist():
            raise RuntimeError("Proved score disagrees with reference")
        entry = {"columns": cols, "implementation": implementation, "k": k,
                 "summary": summary(report), "report": report, "capacity_failures": capacity_errors}
        candidates.append(entry)
        save_json(ROOT / "artifacts/runs/tuning_partial.json", {"manifest": search_manifest, "candidates": candidates})
        print(f"c={cols}, i={implementation}, k={k}: {entry['summary']['median_proving_s']:.4f}s", flush=True)
    tuning_s = time.perf_counter() - search_start
    baseline = min((r for r in candidates if r["columns"] == cfg["baseline_columns"]), key=lambda r: r["summary"]["median_proving_s"])
    tuned = min(candidates, key=lambda r: r["summary"]["median_proving_s"])
    tuning = {"method": "Bounded empirical tuning; NOT the original ZKML cost-estimator optimizer",
              "manifest": search_manifest, "tuning_wall_s": tuning_s,
              "baseline_choice": {k: baseline[k] for k in ("columns", "implementation", "k")},
              "tuned_choice": {k: tuned[k] for k in ("columns", "implementation", "k")}, "candidates": candidates}
    save_json(ROOT / "results/tuning.json", tuning)
    if args.tune_only:
        return
    data = np.load(ROOT / "data/processed/test.npz")
    idx = np.random.default_rng(cfg["seed"] + 1).choice(len(data["x"]), cfg["benchmark_observations"], replace=False)
    x = data["x"][idx]
    reports = {"fixed_40": [], "tuned": []}
    # Alternate configuration order across two blocks; keys reused within each block.
    for block, order in enumerate((("fixed_40", "tuned"), ("tuned", "fixed_40"))):
        for name in order:
            choice = baseline if name == "fixed_40" else tuned
            folder = ROOT / f"artifacts/runs/final_{name}_{block}"
            path = folder / "model.msgpack"
            variant = write_model(model, path, **{k: choice[k] for k in ("columns", "k", "implementation")})
            x_block = np.array_split(x, 2)[block]
            inputs = write_inputs(variant, x_block, folder / "inputs")
            report = run("prove", path, inputs, folder / "proof", timeout=600)
            expected = reference(variant, x_block)[0].tolist()
            assert [r["score_integer"] for r in report["measurements"]] == expected
            reports[name].append(report)
            print(f"Finished block {block}: {name}", flush=True)
    merged = {}
    for name, blocks in reports.items():
        combined = {**blocks[0], "measurements": [r for b in blocks for r in b["measurements"]],
                    "peak_process_rss_bytes": max(b["peak_process_rss_bytes"] for b in blocks)}
        merged[name] = summary(combined)
    fixed_s = merged["fixed_40"]["median_proving_s"]
    tuned_s = merged["tuned"]["median_proving_s"]
    result = {"method": tuning["method"], "test_row_ids": data["row_id"][idx].tolist(),
              "platform": platform.platform(), "threads": cfg["rayon_threads"],
              "binary_sha256": sha256(BINARY), "summary": merged, "raw_blocks": reports,
              "speedup": fixed_s / tuned_s, "latency_reduction": 1 - tuned_s / fixed_s,
              "tuning_wall_s": tuning_s, "tuning_manifest": search_manifest,
              "baseline_choice": tuning["baseline_choice"], "tuned_choice": tuning["tuned_choice"]}
    save_json(ROOT / "results/benchmark.json", result)
    print(json.dumps({"summary": merged, "speedup": result["speedup"]}, indent=2))


if __name__ == "__main__":
    main()
