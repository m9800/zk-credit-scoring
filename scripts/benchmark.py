"""Measure held-out proofs AFTER the original cost estimator freezes both choices."""
import json
import platform
import numpy as np
from scripts.common import ROOT, config, save_json, sha256
from scripts.zk import write_inputs, run, reference, BINARY
import msgpack


def summary(report):
    p = np.array([m["proving_s"] for m in report["measurements"]])
    v = np.array([m["verification_s"] for m in report["measurements"]])
    return {"median_proving_s": float(np.median(p)), "proving_iqr_s": float(np.quantile(p, .75) - np.quantile(p, .25)),
            "median_verification_s": float(np.median(v)), "proof_bytes": report["measurements"][0]["proof_bytes"],
            "peak_process_rss_bytes": report["peak_process_rss_bytes"]}


def main():
    cfg = config()
    optimization = json.loads((ROOT / "results/optimizer.json").read_text())
    if optimization["selection_uses_real_proof_timings"]:
        raise ValueError("This benchmark requires estimator-selected configurations")
    if optimization["tflite_sha256"] != sha256(ROOT / "artifacts/model/model.tflite"):
        raise ValueError("The model changed after optimization")
    if optimization["rayon_threads"] != cfg["rayon_threads"]:
        raise ValueError("Thread count changed after optimization")
    if optimization["preprocessing_sha256"] != sha256(ROOT / "artifacts/model/preprocessing.json"):
        raise ValueError("Preprocessing changed after optimization")
    if optimization["adapter_sha256"] != sha256(ROOT / "build.rs"):
        raise ValueError("The estimator adapter changed after optimization")
    if optimization["calibration_sha256"] != sha256(ROOT / "results/calibration/summary.msgpack"):
        raise ValueError("Calibration changed after optimization")
    quantization = json.loads((ROOT / "artifacts/model/quantization.json").read_text())
    if optimization["scale_factor"] != quantization["scale_factor"]:
        raise ValueError("Scale changed after optimization")
    choices = {"fixed_40": optimization["baseline_choice"], "optimized": optimization["optimized_choice"]}
    model_paths = {"fixed_40": ROOT / "artifacts/model/credit_fixed_40.msgpack",
                   "optimized": ROOT / "artifacts/model/credit_optimized.msgpack"}
    for name, path in model_paths.items():
        if sha256(path) != choices[name]["model_sha256"]:
            raise ValueError("Frozen configuration checksum mismatch")
    data = np.load(ROOT / "data/processed/test.npz")
    idx = np.random.default_rng(cfg["seed"] + 1).choice(len(data["x"]), cfg["benchmark_observations"], replace=False)
    x = data["x"][idx]
    reports = {"fixed_40": [], "optimized": []}
    # Alternate configuration order across two blocks; keys reused within each block.
    for block, order in enumerate((("fixed_40", "optimized"), ("optimized", "fixed_40"))):
        for name in order:
            folder = ROOT / f"artifacts/runs/original_optimizer_{name}_{block}"
            path = model_paths[name]
            variant = msgpack.unpackb(path.read_bytes())
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
    optimized_s = merged["optimized"]["median_proving_s"]
    result = {"method": optimization["method"], "test_row_ids": data["row_id"][idx].tolist(),
              "platform": platform.platform(), "threads": cfg["rayon_threads"],
              "binary_sha256": sha256(BINARY), "summary": merged, "raw_blocks": reports,
              "speedup": fixed_s / optimized_s, "latency_reduction": 1 - optimized_s / fixed_s,
              "optimizer_wall_s": optimization["optimizer_wall_s"],
              "calibration_wall_s": optimization["calibration_wall_s"],
              "optimizer_result_sha256": sha256(ROOT / "results/optimizer.json"),
              "baseline_choice": choices["fixed_40"], "optimized_choice": choices["optimized"]}
    save_json(ROOT / "results/benchmark.json", result)
    print(json.dumps({"summary": merged, "speedup": result["speedup"]}, indent=2))


if __name__ == "__main__":
    main()
