"""Measure held-out proofs AFTER the original cost estimator freezes both choices."""
import json
import argparse
import platform
import numpy as np
from scripts.common import ROOT, config, save_json, sha256
from scripts.zk import write_inputs, run, reference, BINARY
import msgpack


def summary(report):
    p = np.array([m["proving_s"] for m in report["measurements"]])
    v = np.array([m["verification_s"] for m in report["measurements"]])
    return {"median_proving_s": float(np.median(p)), "proving_iqr_s": float(np.quantile(p, .75) - np.quantile(p, .25)),
            "max_proving_s": float(p.max()), "observed_within_15s": bool(p.max() <= 15),
            "median_verification_s": float(np.median(v)), "proof_bytes": report["measurements"][0]["proof_bytes"],
            "peak_process_rss_bytes": report["peak_process_rss_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("logistic", "neural"), default="logistic")
    neural = parser.parse_args().model == "neural"
    cfg = config()
    model_dir = ROOT / ("artifacts/comparison/neural" if neural else "artifacts/model")
    data_dir = ROOT / ("artifacts/comparison" if neural else "data/processed")
    optimization_path = model_dir / "optimizer.json" if neural else ROOT / "results/optimizer.json"
    report_path = model_dir / "benchmark.json" if neural else ROOT / "results/benchmark.json"

    # Check that model and data still match the frozen optimizer result.
    optimization = json.loads(optimization_path.read_text())
    if optimization["selection_uses_real_proof_timings"]:
        raise ValueError("This benchmark requires estimator-selected configurations")
    if neural:
        if optimization["identity"][1] != cfg:
            raise ValueError("Settings changed after optimization")
        for path, digest in optimization["identity"][3]:
            if sha256(ROOT / path) != digest:
                raise ValueError(f"Search input changed after optimization: {path}")
        comparison = json.loads((data_dir / "comparison.json").read_text())
        if comparison["split_sha256"]["test"] != sha256(data_dir / "test.npz"):
            raise ValueError("Test split changed since training")
    else:
        for key, path in (("tflite_sha256", model_dir / "model.tflite"),
                          ("preprocessing_sha256", model_dir / "preprocessing.json"),
                          ("adapter_sha256", ROOT / "build.rs"),
                          ("calibration_sha256", ROOT / "results/calibration/summary.msgpack")):
            if optimization[key] != sha256(path):
                raise ValueError(f"Input changed after optimization: {path}")
    if optimization["rayon_threads"] != cfg["rayon_threads"]:
        raise ValueError("Thread count changed after optimization")
    quantization = json.loads((model_dir / "quantization.json").read_text())
    if optimization["scale_factor"] != quantization["scale_factor"]:
        raise ValueError("Scale changed after optimization")

    choices = {"fixed_40": optimization["baseline_choice"], "optimized": optimization["optimized_choice"]}
    model_paths = {name: model_dir / f"credit_{name}.msgpack" for name in choices}
    for name, path in model_paths.items():
        if sha256(path) != choices[name]["model_sha256"]:
            raise ValueError("Frozen configuration checksum mismatch")

    # Choose observations only after both circuit layouts are frozen.
    data = np.load(data_dir / "test.npz")
    idx = np.random.default_rng(cfg["seed"] + 1).choice(len(data["x"]), cfg["benchmark_observations"], replace=False)
    x = data["x"][idx]
    if len(x) < 2:
        raise ValueError("Two proof blocks require at least two observations")
    reports = {"fixed_40": [], "optimized": []}
    scores = {}
    # Alternate configuration order across two blocks; keys reused within each block.
    for block, order in enumerate((("fixed_40", "optimized"), ("optimized", "fixed_40"))):
        for name in order:
            folder = (model_dir / "benchmark" / f"{name}_{block}" if neural else
                      ROOT / f"artifacts/runs/original_optimizer_{name}_{block}")
            path = model_paths[name]
            variant = msgpack.unpackb(path.read_bytes())
            x_block = np.array_split(x, 2)[block]
            inputs = write_inputs(variant, x_block, folder / "inputs")
            report = run("prove", path, inputs, folder / "proof", timeout=600)
            expected = reference(variant, x_block)[0].tolist()
            if (block in scores and scores[block] != expected) or [r["score_integer"] for r in report["measurements"]] != expected:
                raise ValueError("Circuit scores disagree across layouts or with the reference")
            scores[block] = expected
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
              "optimizer_wall_s": optimization["invocation_wall_s" if neural else "optimizer_wall_s"],
              "calibration_wall_s": optimization["calibration_wall_s"],
              "optimizer_result_sha256": sha256(optimization_path),
              "baseline_choice": choices["fixed_40"], "optimized_choice": choices["optimized"]}
    save_json(report_path, result)
    print(json.dumps({"summary": merged, "speedup": result["speedup"]}, indent=2))


if __name__ == "__main__":
    main()
