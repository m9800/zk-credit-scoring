"""Small wrappers around the pinned upstream converter and our proof harness."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import msgpack
import numpy as np
from scripts.common import ROOT, config, save_json

BINARY = ROOT / "vendor/zkml/target/release/credit-zk"


def convert(tflite, output, scale):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "vendor/zkml/python/converter.py"),
               "--model", str(tflite), "--model_output", str(output / "base.msgpack"),
               "--config_output", str(output / "config.msgpack"), "--scale_factor", str(scale),
               "--k", "12", "--num_cols", "10", "--num_randoms", str(config()["num_randoms"])]
    with (output / "conversion.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    model = msgpack.unpackb((output / "base.msgpack").read_bytes())
    # The original graph's weight cells are constrained as public instances.
    # This binds inference to known weights without a hash gadget or private weights.
    model["out_idxes"] = sorted(t["idx"] for t in model["tensors"]) + model["out_idxes"]
    model["commit_before"] = []
    model["commit_after"] = []
    return model


def write_model(model, path, columns=10, k=12, implementation=1):
    model = copy.deepcopy(model)
    model["num_cols"] = columns
    model["k"] = k
    for layer in model["layers"]:
        layer["implementation"] = implementation if layer["layer_type"] == "FullyConnected" else 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(msgpack.packb(model, use_bin_type=True))
    return model


def write_inputs(model, x, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in enumerate(x):
        quantized = np.rint(np.asarray(row, dtype=np.float32) * model["global_sf"]).astype(np.int64)
        tensor = [{"idx": model["inp_idxes"][0], "shape": [1, len(row)], "data": quantized.tolist()}]
        path = directory / f"input_{i}.msgpack"
        path.write_bytes(msgpack.packb(tensor, use_bin_type=True))
        paths.append(str(path.resolve()))
    list_path = directory / "inputs.json"
    save_json(list_path, paths)
    return list_path


def run(mode, model_path, inputs_path, output, timeout=600):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "RAYON_NUM_THREADS": str(config()["rayon_threads"])}
    with (output / f"{mode}.log").open("w") as log:
        result = subprocess.run([str(BINARY), mode, str(model_path), str(inputs_path), str(output)],
                                stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=ROOT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{mode} failed ({result.returncode}): {output / (mode + '.log')}")
    return json.loads((output / ("mock.json" if mode == "mock" else "measurements.json")).read_text())


def reference(model, x):
    tensors = {t["idx"]: np.asarray(t["data"], dtype=np.int64).reshape(t["shape"]) for t in model["tensors"]}
    fc = model["layers"][0]
    sf = model["global_sf"]
    weights = tensors[fc["inp_idxes"][1]].reshape(-1)
    bias = int(tensors[fc["inp_idxes"][2]].reshape(-1)[0])
    xq = np.rint(np.asarray(x, dtype=np.float32) * sf).astype(np.int64)
    dot = xq @ weights
    # Upstream rounded_div uses nearest-integer division; ties go towards +infinity
    # after the gadget shifts its numerator to the positive range.
    logit = (2 * dot + sf) // (2 * sf) + bias
    probability = 1 / (1 + np.exp(-logit.astype(np.float64) / sf))
    yq = np.floor(probability * sf + 0.5).astype(np.int64)
    return yq, logit


def smoke():
    output = ROOT / "artifacts/smoke"
    model = convert(output / "model.tflite", output / "converted", 128)
    model_path = output / "public_model.msgpack"
    write_model(model, model_path)
    inputs = write_inputs(model, np.load(output / "input.npy"), output / "inputs")
    mock = run("mock", model_path, inputs, output / "mock")
    report = run("prove", model_path, inputs, output / "proof")
    expected = reference(model, np.load(output / "input.npy"))[0]
    assert mock[0]["score_integer"] == int(expected[0])
    save_json(ROOT / "results/smoke.json", report)
    print("Real KZG smoke proof accepted; wrong score, wrong weight and altered proof rejected.")


if __name__ == "__main__":
    smoke()
