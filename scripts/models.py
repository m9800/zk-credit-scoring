"""The 4→64→32→1 neural candidate, compared with the saved logistic baseline.

It consumes the baseline's four preprocessed features. Model fitting is separate
from circuit layout: ZKML implementation choices never change these weights.
"""

import copy
import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class NeuralTraining:
    seed: int = 42
    max_epochs: int = 60
    patience: int = 10
    min_improvement: float = 0.0001

    def __post_init__(self):
        if (
            type(self.max_epochs) is not int
            or self.max_epochs < 1
            or type(self.patience) is not int
            or self.patience < 1
            or not math.isfinite(self.min_improvement)
            or not 0 <= self.min_improvement <= 1
        ):
            raise ValueError(
                "Epochs/patience must be positive integers; improvement must be finite in [0, 1]"
            )


def neural_network(seed=42):
    """2,433 parameters: (4*64+64) + (64*32+32) + (32+1).

    Binary MLPClassifier uses a sigmoid output; its hidden layers use ReLU.
    partial_fit below gives explicit validation-AP checkpoint selection.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1.0,
        batch_size=1024,
        learning_rate_init=0.001,
        random_state=seed,
        shuffle=True,
    )


def fit_network(x, y, validation_x, validation_y, training=NeuralTraining()):
    """Select checkpoints exclusively by validation AP; no test inputs accepted."""
    model = neural_network(training.seed)
    best, best_ap, best_epoch = None, -1.0, 0
    history = []

    for epoch in range(1, training.max_epochs + 1):
        model.partial_fit(x, y, classes=np.array([0, 1]))
        ap = float(
            average_precision_score(
                validation_y, model.predict_proba(validation_x)[:, 1]
            )
        )
        history.append({"epoch": epoch, "average_precision": ap})

        if best is None or ap > best_ap + training.min_improvement:
            best = copy.deepcopy(model)
            best_ap = ap
            best_epoch = epoch

        if epoch - best_epoch >= training.patience:
            break

    return best, {"selected_epoch": best_epoch, "epochs_run": epoch, "history": history}


def keras_model(fitted):
    """Transfer learned weights to the equivalent graph for TFLite/ZKML export."""
    import tensorflow as tf

    if fitted.n_features_in_ != 4 or list(fitted.classes_) != [0, 1]:
        raise ValueError("Expected four features and binary classes [0, 1]")
    if (
        not isinstance(fitted, MLPClassifier)
        or fitted.hidden_layer_sizes != (64, 32)
        or fitted.activation != "relu"
    ):
        raise ValueError("Expected the 64/32 ReLU neural candidate")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(batch_shape=(1, 4)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    weights = [v for pair in zip(fitted.coefs_, fitted.intercepts_) for v in pair]
    model.set_weights([w.astype(np.float32) for w in weights])
    return model
