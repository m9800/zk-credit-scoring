"""Fit two tiny logistic candidates on chronological training data only."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from scripts.common import ROOT, config, save_json, sha256


def metrics(y, p):
    return {"average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "positive_rate": float(np.mean(y)), "n": len(y)}


def transform(frame, state):
    values = frame[state["features"]].to_numpy(dtype=np.float64).copy()
    values[~np.isfinite(values)] = np.nan
    for i, name in enumerate(state["features"]):
        if name in state["log_features"]:
            if np.any(values[:, i] < 0):
                raise ValueError(f"Negative values in log feature {name}")
            values[:, i] = np.log1p(values[:, i])
    values = np.where(np.isnan(values), state["medians"], values)
    values = np.clip(values, state["lower"], state["upper"])
    return ((values - state["means"]) / state["stds"]).astype(np.float32)


def fit_preprocessing(frame, features, log_features):
    values = frame[features].to_numpy(dtype=np.float64).copy()
    values[~np.isfinite(values)] = np.nan
    for i, name in enumerate(features):
        if name in log_features:
            if np.any(values[:, i] < 0):
                raise ValueError(f"Negative values in log feature {name}")
            values[:, i] = np.log1p(values[:, i])
    medians = np.nanmedian(values, axis=0)
    if not np.all(np.isfinite(medians)):
        raise ValueError("A selected feature has no finite training observations")
    values = np.where(np.isnan(values), medians, values)
    lower, upper = np.quantile(values, [0.005, 0.995], axis=0)
    values = np.clip(values, lower, upper)
    stds = np.std(values, axis=0)
    stds[stds == 0] = 1
    return {"features": features, "log_features": log_features,
            "medians": medians.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
            "means": values.mean(axis=0).tolist(), "stds": stds.tolist()}


def split_masks(timestamps, embargo_days):
    ordered = np.sort(np.asarray(timestamps))
    val_start = float(ordered[int(len(ordered) * 0.70)])
    test_start = float(ordered[int(len(ordered) * 0.85)])
    gap = embargo_days * 86400
    timestamps = np.asarray(timestamps)
    masks = {"train": timestamps < val_start - gap,
             "validation": (timestamps >= val_start) & (timestamps < test_start - gap),
             "test": timestamps >= test_start}
    return masks, {"validation_start": val_start, "test_start": test_start,
                   "embargo_days": embargo_days}


def main():
    cfg = config()
    dataset = ROOT / "data/raw/spectral.parquet"
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if sha256(dataset) != manifest["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    df = pd.read_parquet(dataset)
    df = df.sort_values(["borrow_timestamp", "borrow_block_number", "wallet_address"], kind="stable")
    original_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if not set(df.target.unique()) <= {0, 1} or df.borrow_timestamp.isna().any():
        raise ValueError("Invalid target or timestamp")
    masks, boundaries = split_masks(df.borrow_timestamp, cfg["embargo_days"])
    parts = {name: df.loc[mask] for name, mask in masks.items()}
    for name, frame in parts.items():
        if frame.target.nunique() != 2:
            raise ValueError(f"{name} does not contain both classes")
    audit = {"rows_raw": original_count, "rows_deduplicated": len(df),
             "unique_wallets": int(df.wallet_address.nunique()), "columns": list(df.columns),
             "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
             "start_utc": pd.to_datetime(df.borrow_timestamp.min(), unit="s", utc=True).isoformat(),
             "end_utc": pd.to_datetime(df.borrow_timestamp.max(), unit="s", utc=True).isoformat(),
             "boundaries": boundaries, "embargo_status": cfg["embargo_status"],
             "removed_by_embargo": int(len(df) - sum(len(v) for v in parts.values())),
             "splits": {k: {"rows": len(v), "positives": int(v.target.sum()),
                            "wallets": int(v.wallet_address.nunique())} for k, v in parts.items()},
             "train_test_shared_wallets": len(set(parts["train"].wallet_address) & set(parts["test"].wallet_address))}
    save_json(ROOT / "results/data_audit.json", audit)
    candidates = []
    for count in (4, 8):
        features = cfg[f"features_{count}"]
        state = fit_preprocessing(parts["train"], features, cfg["log_features"])
        x_train = transform(parts["train"], state)
        x_val = transform(parts["validation"], state)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=cfg["seed"])
        model.fit(x_train, parts["train"].target)
        if model.n_iter_.max() >= model.max_iter:
            raise RuntimeError("Logistic regression did not converge")
        score = metrics(parts["validation"].target.to_numpy(), model.predict_proba(x_val)[:, 1])
        candidates.append((count, state, model, score))
        print(f"{count} features: {score}", flush=True)
    best_ap = max(c[3]["average_precision"] for c in candidates)
    chosen = next(c for c in candidates if c[3]["average_precision"] >= best_ap - 0.01)
    count, state, model, score = chosen
    output = ROOT / "artifacts/model"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "preprocessing.json", state)
    save_json(output / "weights.json", {"weights": model.coef_[0].tolist(), "bias": float(model.intercept_[0])})
    save_json(ROOT / "results/training.json", {
        "selected_features": count, "candidates": [{"features": c[0], "validation": c[3]} for c in candidates],
        "selection_rule": "Smallest feature set within 0.01 validation average precision of best",
        "training": "scikit-learn LogisticRegression, C=1, LBFGS; transferred to equivalent Keras Dense",
        "constant_validation": metrics(parts["validation"].target.to_numpy(),
            np.full(len(parts["validation"]), parts["train"].target.mean())),
        "training_positive_rate": float(parts["train"].target.mean()),
        "test_evaluated": False})
    import tensorflow as tf
    from scripts.smoke_model import export
    keras = tf.keras.Sequential([tf.keras.layers.Input(batch_shape=(1, count)),
                                tf.keras.layers.Dense(1, activation="sigmoid")])
    keras.set_weights([model.coef_.T.astype(np.float32), model.intercept_.astype(np.float32)])
    keras.save(output / "model.keras")
    operations = export(keras, output / "model.tflite")
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "test"):
        part = parts[name]
        np.savez_compressed(processed / f"{name}.npz", x=transform(part, state),
                            y=part.target.to_numpy(np.int64), row_id=part.index.to_numpy())
    val = transform(parts["validation"], state)
    np.save(output / "input.npy", val[:1])
    save_json(output / "manifest.json", {"operations": operations, "tflite_sha256": sha256(output / "model.tflite"),
              "preprocessing_sha256": sha256(output / "preprocessing.json"), "seed": cfg["seed"]})
    print(f"Selected {count} features; held-out test predictions have not been evaluated.")


if __name__ == "__main__":
    main()
