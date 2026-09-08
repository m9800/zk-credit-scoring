"""Select fixed-point scale on validation only, cross-checking actual circuits."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json
import numpy as np
import tensorflow as tf
from scripts.common import ROOT, config, save_json
from scripts.train import metrics
from scripts.zk import convert, write_model, write_inputs, run, reference


def tflite_predict(path, x):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]["index"]
    out = interpreter.get_output_details()[0]["index"]
    probabilities = []
    for row in x:
        interpreter.set_tensor(inp, row.reshape(1, -1))
        interpreter.invoke()
        probabilities.append(float(interpreter.get_tensor(out)[0, 0]))
    return np.asarray(probabilities)


def main():
    cfg = config()
    data = np.load(ROOT / "data/processed/validation.npz")
    x, y = data["x"], data["y"]
    model_path = ROOT / "artifacts/model/model.tflite"
    fp = tflite_predict(model_path, x)
    keras = tf.keras.models.load_model(ROOT / "artifacts/model/model.keras")
    keras_fp = keras(x, training=False).numpy().reshape(-1)
    tf_error = float(np.max(np.abs(fp - keras_fp)))
    assert tf_error < 1e-5
    fp_metrics = metrics(y, fp)
    rows = []
    selected = None
    for sf in cfg["scale_candidates"]:
        folder = ROOT / f"artifacts/zk/scale_{sf}"
        model = convert(model_path, folder, sf)
        yq, logit = reference(model, x)
        quantized = yq / sf
        quant_metrics = metrics(y, quantized)
        disagreement = float(np.mean((fp >= cfg["classification_threshold"]) != (quantized >= cfg["classification_threshold"])))
        error = float(np.max(np.abs(fp - quantized)))
        eligible = (fp_metrics["average_precision"] - quant_metrics["average_precision"] < cfg["max_ap_degradation"]
                    and disagreement < cfg["max_disagreement"] and error <= cfg["max_probability_error"])
        row = {"scale_factor": sf, "metrics": quant_metrics, "classification_disagreement": disagreement,
               "max_probability_error": error, "passes_tolerances": eligible,
               "max_abs_logit_integer": int(np.abs(logit).max())}
        rows.append(row)
        print(row, flush=True)
        if eligible and selected is None:
            selected = (sf, model, yq, logit)
    if selected is None:
        save_json(ROOT / "results/validation.json", {"scales": rows, "selected": None})
        raise RuntimeError("No scale passed the predeclared validation tolerances")
    sf, model, yq, logit = selected
    # Include extrema, near-threshold values and a seeded uniform validation sample.
    rng = np.random.default_rng(cfg["seed"])
    idx = np.unique(np.r_[np.argmin(logit), np.argmax(logit),
        np.argsort(np.abs(fp - cfg["classification_threshold"]))[:4], rng.choice(len(x), 10, replace=False)])
    folder = ROOT / "artifacts/zk/validation"
    # k affects lookup ranges as well as capacity. Start large enough for all validation logits.
    minimum_k = max(8, int(np.ceil(np.log2(max(2 * (int(np.abs(logit).max()) + 10), 2 * sf + 20)))))
    path = folder / "model.msgpack"
    write_model(model, path, k=max(12, minimum_k))
    inputs = write_inputs(model, x[idx], folder / "inputs")
    actual = run("mock", path, inputs, folder / "mock")
    assert [r["score_integer"] for r in actual] == yq[idx].tolist(), "Reference disagrees with upstream circuit"
    save_json(ROOT / "artifacts/model/quantization.json", {"scale_factor": sf, "minimum_k_from_ranges": minimum_k,
              "reference_crosscheck_observations": len(idx), "classification_threshold": cfg["classification_threshold"]})
    save_json(ROOT / "results/validation.json", {"keras_tflite_max_error": tf_error, "tflite_metrics": fp_metrics,
              "scales": rows, "selected_scale": sf, "mock_crosscheck_count": len(idx),
              "reference_matches_mock": True, "test_used_for_selection": False})
    print(f"Selected scale {sf}; reference matched {len(idx)} actual mock circuits.")


if __name__ == "__main__":
    main()
