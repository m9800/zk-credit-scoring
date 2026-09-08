"""Run the authors' KZG microbenchmarks and retain Criterion's raw samples."""
import json
import os
import platform
import subprocess
import time
import msgpack
from scripts.common import ROOT, config, save_json, sha256

OPERATIONS = ("fft", "msm", "add", "mul", "permute")
CALIBRATION_K = range(10, 15)


def main():
    output = ROOT / "results/calibration"
    output.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "artifacts/optimizer/calibration_logs"
    logs.mkdir(parents=True, exist_ok=True)
    criterion_home = ROOT / "artifacts/optimizer/criterion"
    env = {**os.environ, "RAYON_NUM_THREADS": str(config()["rayon_threads"]),
           "CRITERION_HOME": str(criterion_home)}
    subprocess.run(["cargo", "bench", "--no-run", "--locked"], cwd=ROOT, env=env, check=True)
    start = time.perf_counter()
    summary = {}
    for operation in OPERATIONS:
        command = ["cargo", "bench", "--locked", "--bench", f"calibrate_{operation}", "--",
                   "--warm-up-time", "0.5", "--measurement-time", "2", "--sample-size", "30", "--noplot"]
        with (logs / f"{operation}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        coefficients = {}
        for k in CALIBRATION_K:
            directory = criterion_home / f"kzg_{operation}/k/{k}/new"
            estimates = json.loads((directory / "estimates.json").read_text())
            sample = json.loads((directory / "sample.json").read_text())
            coefficients[str(k)] = estimates["mean"]["point_estimate"]
            save_json(output / "raw" / f"{operation}_{k}.json", {"estimates": estimates, "sample": sample})
        summary[f"kzg_{operation}"] = coefficients
        print(f"Calibrated {operation}: k=10..14", flush=True)
    save_json(output / "summary.json", summary)
    (output / "summary.msgpack").write_bytes(msgpack.packb(summary, use_bin_type=True))
    metadata = {"method": "Original ZKML Criterion KZG microbenchmarks; range adapted from 13..19 to 10..14",
                "calibration_wall_s": time.perf_counter() - start, "compilation_excluded": True,
                "warmup_seconds": 0.5, "measurement_seconds": 2, "samples": 30,
                "rayon_threads": config()["rayon_threads"], "platform": platform.platform(),
                "rustc": subprocess.check_output(["rustc", "--version"], cwd=ROOT, text=True).strip(),
                "summary_sha256": sha256(output / "summary.msgpack"),
                "adapter_sha256": sha256(ROOT / "build.rs")}
    save_json(output / "metadata.json", metadata)
    print(f"Calibration complete in {metadata['calibration_wall_s']:.1f}s (excluding compilation).")


if __name__ == "__main__":
    main()
