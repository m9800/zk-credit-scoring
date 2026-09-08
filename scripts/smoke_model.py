"""Export a deterministic tiny model to test ZKML before training."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from scripts.common import ROOT, save_json


def export(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    path.write_bytes(converter.convert())
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    operations = [op["op_name"] for op in interpreter._get_ops_details()
                  if op["op_name"] != "DELEGATE"]
    if operations != ["FULLY_CONNECTED", "LOGISTIC"]:
        raise ValueError(f"Unexpected TFLite operations: {operations}")
    return operations


def main():
    output = ROOT / "artifacts/smoke"
    output.mkdir(parents=True, exist_ok=True)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(1, 4)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.set_weights([np.array([[0.25], [-0.5], [0.125], [0.25]], np.float32),
                       np.array([0.125], np.float32)])
    x = np.array([[0.5, 0.25, -0.5, 1.0]], np.float32)
    np.save(output / "input.npy", x)
    operations = export(model, output / "model.tflite")
    save_json(output / "expected.json", {"probability": float(model(x)[0, 0]), "operations": operations})
    print("Smoke model exported:", output)


if __name__ == "__main__":
    main()
