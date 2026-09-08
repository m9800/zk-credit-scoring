"""Evaluate held-out data only after model and quantization are frozen."""
import json
import time
import numpy as np
from scripts.common import ROOT, config, save_json
from scripts.train import metrics
from scripts.zk import convert, reference
from scripts.validate import tflite_predict


def calibration(y, p):
    rows = []
    for lower in np.arange(0, 1, 0.1):
        selected = (p >= lower) & (p < lower + 0.1 if lower < 0.9 else p <= 1)
        if selected.any():
            rows.append({"lower": float(lower), "n": int(selected.sum()),
                         "mean_prediction": float(p[selected].mean()), "positive_rate": float(y[selected].mean())})
    return rows


def main():
    quantization = json.loads((ROOT / "artifacts/model/quantization.json").read_text())
    sf = quantization["scale_factor"]
    data = np.load(ROOT / "data/processed/test.npz")
    x, y = data["x"], data["y"]
    path = ROOT / "artifacts/model/model.tflite"
    fp = tflite_predict(path, x)
    model = convert(path, ROOT / "artifacts/zk/test_reference", sf)
    quantized = reference(model, x)[0] / sf
    training = json.loads((ROOT / "results/training.json").read_text())
    constant = np.full(len(y), training["training_positive_rate"])
    # Preallocate the interpreter and input tensor; exclude loading from timing.
    import tensorflow as tf
    keras = tf.keras.models.load_model(ROOT / "artifacts/model/model.keras")
    keras_fp = keras(x, training=False).numpy().reshape(-1)
    keras_tflite_error = float(np.max(np.abs(keras_fp - fp)))
    assert keras_tflite_error < 1e-5
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]["index"]
    out = interpreter.get_output_details()[0]["index"]
    latencies = []
    for i in range(110):
        value = x[i].reshape(1, -1)
        start = time.perf_counter_ns()
        interpreter.set_tensor(inp, value)
        interpreter.invoke()
        interpreter.get_tensor(out)
        elapsed = time.perf_counter_ns() - start
        if i >= 10:
            latencies.append(elapsed)
    report = {"test_tflite": metrics(y, fp), "test_fixed_point_reference": metrics(y, quantized),
              "keras_tflite_max_error": keras_tflite_error,
              "test_constant": metrics(y, constant), "scale_factor": sf,
              "max_probability_error": float(np.max(np.abs(fp - quantized))),
              "classification_disagreement": float(np.mean((fp >= 0.5) != (quantized >= 0.5))),
              "calibration": calibration(y, quantized), "tflite_latency_ns": latencies,
              "tflite_latency_median_ns": float(np.median(latencies)),
              "fixed_point_method": "Vectorized reference cross-checked against actual upstream mock circuits; not a proof for every test row"}
    save_json(ROOT / "results/test_metrics.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("calibration", "tflite_latency_ns")}, indent=2))


if __name__ == "__main__":
    main()
