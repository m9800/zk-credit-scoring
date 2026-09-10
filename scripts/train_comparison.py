"""Train/export the neural candidate against the existing frozen logistic baseline.

Exports the NN separately and leaves the saved logistic baseline unchanged.
Test predictions are computed only after both models/checkpoints are frozen.
"""

import argparse
from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from scripts.common import ROOT, config, save_json, sha256
from scripts.models import NeuralTraining, fit_network, keras_model
from scripts.smoke_model import export
from scripts.train import transform, split_masks, metrics


def load_shared_data(cfg):
    """Reconstruct the original splits using the saved baseline preprocessing."""
    source = ROOT / "data/raw/spectral.parquet"
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if sha256(source) != manifest["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    frame = (
        pd.read_parquet(source)
        .sort_values(
            ["borrow_timestamp", "borrow_block_number", "wallet_address"], kind="stable"
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if frame.borrow_timestamp.isna().any() or not set(frame.target.unique()) <= {0, 1}:
        raise ValueError("Invalid labels or timestamps")

    masks, boundaries = split_masks(frame.borrow_timestamp, cfg["embargo_days"])
    for name, mask in masks.items():
        if frame.loc[mask, "target"].nunique() != 2:
            raise ValueError(f"{name} must contain both labels")

    # Load the saved baseline and its frozen preprocessing.
    baseline_dir = ROOT / "artifacts/model"
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text())
    for name in ("model.tflite", "preprocessing.json"):
        key = "tflite_sha256" if name == "model.tflite" else "preprocessing_sha256"
        if sha256(baseline_dir / name) != baseline_manifest[key]:
            raise ValueError(f"Existing baseline artifact changed: {name}")

    preprocessing = json.loads((baseline_dir / "preprocessing.json").read_text())
    if preprocessing["features"] != cfg["features_4"]:
        raise ValueError("The saved baseline must use the selected four features")

    audit = json.loads((ROOT / "results/data_audit.json").read_text())
    if boundaries != audit["boundaries"] or any(
        int(mask.sum()) != audit["splits"][name]["rows"] for name, mask in masks.items()
    ):
        raise ValueError("Splits differ from the existing baseline")

    parts = {
        name: (
            transform(frame.loc[mask], preprocessing),
            frame.loc[mask, "target"].to_numpy(np.int64),
        )
        for name, mask in masks.items()
    }
    return frame, masks, boundaries, manifest, preprocessing, parts


def baseline_metrics(parts):
    """Evaluate the exact saved TFLite model without retraining or exporting it."""
    from scripts.validate import tflite_predict

    baseline_dir = ROOT / "artifacts/model"
    report = {
        "parameters": 5,
        "reused_model": "artifacts/model/model.tflite",
        "tflite_sha256": sha256(baseline_dir / "model.tflite"),
        "preprocessing_sha256": sha256(baseline_dir / "preprocessing.json"),
    }
    for name in ("validation", "test"):
        x, y = parts[name]
        report[name] = metrics(y, tflite_predict(baseline_dir / "model.tflite", x))
    return report


def export_candidate(candidate, parts, directory, seed):
    """Export the neural model and check prediction parity across conversions."""
    from scripts.validate import tflite_predict

    validation_x, validation_y = parts["validation"]
    test_x, test_y = parts["test"]
    directory.mkdir(parents=True, exist_ok=True)
    keras = keras_model(candidate)
    transferred = keras(validation_x, training=False).numpy().reshape(-1)
    expected = candidate.predict_proba(validation_x)[:, 1]
    transfer_error = float(np.max(np.abs(transferred - expected)))
    if transfer_error >= 1e-5:
        raise ValueError("Neural sklearn/Keras transfer mismatch")
    keras.save(directory / "model.keras")
    operations = export(keras, directory / "model.tflite", fully_connected_layers=3)

    # Compare the exported TFLite graph on near-threshold and seeded rows.
    near_threshold = np.argsort(np.abs(expected - 0.5))[:8]
    random_rows = np.random.default_rng(seed).choice(
        len(validation_x), min(24, len(validation_x)), replace=False
    )
    sample = np.unique(np.r_[near_threshold, random_rows])
    exported_predictions = tflite_predict(
        directory / "model.tflite", validation_x[sample]
    )
    tflite_error = float(np.max(np.abs(exported_predictions - expected[sample])))
    if tflite_error >= 1e-5:
        raise ValueError("Neural TFLite prediction mismatch")

    return {
        "parameters": keras.count_params(),
        "operations": operations,
        "estimator_parameters": candidate.get_params(deep=False),
        "validation": metrics(validation_y, expected),
        "test": metrics(test_y, candidate.predict_proba(test_x)[:, 1]),
        "keras_transfer_max_error": transfer_error,
        "tflite_sample_max_error": tflite_error,
        "tflite_sha256": sha256(directory / "model.tflite"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Neural epoch ceiling; changes the experiment",
    )
    args = parser.parse_args()
    cfg = config()
    training = NeuralTraining(seed=cfg["seed"], max_epochs=args.epochs)
    frame, masks, boundaries, manifest, preprocessing, parts = load_shared_data(cfg)

    candidate, history = fit_network(*parts["train"], *parts["validation"], training)
    output = ROOT / "artifacts/comparison"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "preprocessing.json", preprocessing)
    reports = {
        "logistic": baseline_metrics(parts),
        "neural": export_candidate(candidate, parts, output / "neural", training.seed),
    }

    # Save shared data and the provenance needed by validation and optimization.
    for name in ("validation", "test"):
        np.savez_compressed(
            output / f"{name}.npz",
            x=parts[name][0],
            y=parts[name][1],
            row_id=frame.loc[masks[name]].index.to_numpy(),
        )

    save_json(
        output / "comparison.json",
        {
            "models": reports,
            "training": asdict(training),
            "neural_checkpoint": history,
            "dataset": manifest,
            "boundaries": boundaries,
            "preprocessing_sha256": sha256(output / "preprocessing.json"),
            "split_sha256": {
                name: sha256(output / f"{name}.npz") for name in ("validation", "test")
            },
            "config_sha256": sha256(ROOT / "configs/experiment.json"),
            "source_sha256": {
                name: sha256(ROOT / "scripts" / name)
                for name in (
                    "models.py",
                    "train_comparison.py",
                    "train.py",
                    "smoke_model.py",
                    "common.py",
                )
            },
            "split_rows": {name: len(pair[1]) for name, pair in parts.items()},
            "test_status": "Previously inspected exploratory split; not a fresh final holdout",
            "proof_status": "Training/export only; fresh circuit validation and optimizer runs are separate",
        },
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
