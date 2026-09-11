import numpy as np
import pytest

from scripts.models import NeuralTraining, fit_network, keras_model, neural_network
from scripts.smoke_model import export


def examples():
    x = np.random.default_rng(10).normal(size=(256, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] * x[:, 2] > 0).astype(np.int64)
    return x[:192], y[:192], x[192:], y[192:]


def test_checkpoint_is_the_selected_epoch_not_last_weights():
    x, y, xv, yv = examples()
    # A unit-sized required AP improvement cannot follow the first epoch.
    fitted, report = fit_network(
        x, y, xv, yv, NeuralTraining(max_epochs=5, patience=1, min_improvement=1.0)
    )
    expected = neural_network()
    expected.partial_fit(x, y, classes=np.array([0, 1]))
    assert report["selected_epoch"] == 1 and report["epochs_run"] == 2
    np.testing.assert_array_equal(fitted.predict_proba(xv), expected.predict_proba(xv))


def test_fitted_model_exports_the_same_predictions(tmp_path):
    import tensorflow as tf

    x, y, xv, yv = examples()
    fitted = fit_network(x, y, xv, yv, NeuralTraining(max_epochs=2))[0]
    keras = keras_model(fitted)
    assert keras.count_params() == 2433
    np.testing.assert_allclose(
        keras(xv, training=False).numpy().reshape(-1),
        fitted.predict_proba(xv)[:, 1],
        atol=1e-6,
    )
    path = tmp_path / "model.tflite"
    assert export(keras, path, fully_connected_layers=3) == ["FULLY_CONNECTED"] * 3 + [
        "LOGISTIC"
    ]
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    actual = []
    for row in xv[:8]:
        interpreter.set_tensor(
            interpreter.get_input_details()[0]["index"], row.reshape(1, 4)
        )
        interpreter.invoke()
        actual.append(
            float(
                interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[
                    0, 0
                ]
            )
        )
    np.testing.assert_allclose(actual, fitted.predict_proba(xv[:8])[:, 1], atol=1e-6)


def test_invalid_training_limits():
    with pytest.raises(ValueError):
        NeuralTraining(max_epochs=0)
    with pytest.raises(ValueError):
        NeuralTraining(patience=0)
    with pytest.raises(ValueError):
        NeuralTraining(min_improvement=float("nan"))
    with pytest.raises(ValueError):
        NeuralTraining(min_improvement=2)
