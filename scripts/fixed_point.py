"""Reference for a sequential FC/(fused ReLU) graph ending in scalar logistic.

This is a numerical reference, not a proof. Converter tensor IDs and public
outputs are respected. Actual circuit checks remain required for each model.
Unsupported graphs and potential int64 arithmetic overflow fail explicitly.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Inference:
    """Scores/logits plus a sampled range summary, not a circuit-capacity bound.

    max_abs_intermediate covers quantized inputs and rounded, biased FC outputs
    before ReLU. It excludes raw dot-product accumulators, constants and sigmoid
    scores. Downstream feasibility must check its own complete gadget ranges.
    """

    scores: np.ndarray
    logits: np.ndarray
    max_abs_intermediate: int
    max_abs_by_row: np.ndarray


def _magnitude(values):
    return max(abs(int(values.min())), abs(int(values.max())))


def evaluate(model, x) -> Inference:
    sf = model["global_sf"]
    if type(sf) is not int or not 0 < sf <= 2**31 - 1:
        raise ValueError("Scale must be a positive integer no larger than 2**31 - 1")
    if len(model["inp_idxes"]) != 1:
        raise ValueError("Expected one input tensor")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("Inputs must be a nonempty finite batch matrix")
    quantized = np.rint(x * sf)
    if (
        not np.isfinite(quantized).all()
        or (quantized >= 2**63).any()
        or (quantized < -(2**63)).any()
    ):
        raise ValueError("Quantized inputs exceed int64")
    current = quantized.astype(np.int64)
    current_id = model["inp_idxes"][0]
    tensors = {}
    for tensor in model["tensors"]:
        raw = np.asarray(tensor["data"])
        if raw.dtype.kind not in "iu" or not raw.size or int(raw.max()) >= 2**63:
            raise ValueError("Model constants must be nonempty int64 tensors")
        if tensor["idx"] in tensors or tensor["idx"] == current_id:
            raise ValueError("Duplicate or input-shadowing constant tensor")
        tensors[tensor["idx"]] = raw.astype(np.int64).reshape(tensor["shape"])
    maximum = _magnitude(current)
    by_row = np.max(np.abs(current.astype(object)), axis=1)
    layers = model["layers"]
    if len(layers) < 2 or layers[-1]["layer_type"] != "Logistic":
        raise ValueError("Expected fully-connected layers followed by scalar Logistic")
    used_ids = set(tensors) | {current_id}
    for layer in layers[:-1]:
        if layer["layer_type"] != "FullyConnected":
            raise ValueError("Unsupported intermediate layer")
        ids = layer["inp_idxes"]
        outputs = layer["out_idxes"]
        if (
            len(ids) not in (2, 3)
            or ids[0] != current_id
            or len(outputs) != 1
            or outputs[0] in used_ids
        ):
            raise ValueError(
                "Expected a sequential fully-connected graph with unique outputs"
            )
        weights = tensors[ids[1]]
        if weights.ndim != 2 or current.shape[1] != weights.shape[1]:
            raise ValueError("Fully-connected input/weight dimensions differ")
        bias = (
            tensors[ids[2]].reshape(-1)
            if len(ids) == 3
            else np.zeros(weights.shape[0], dtype=np.int64)
        )
        if bias.shape != (weights.shape[0],):
            raise ValueError("Bias must have one value per output neuron")
        bound = _magnitude(current) * _magnitude(weights) * current.shape[1]
        if (
            bound > np.iinfo(np.int64).max
            or (bound + sf - 1) // sf + _magnitude(bias) > np.iinfo(np.int64).max
        ):
            raise ValueError("Fully-connected arithmetic may overflow int64")
        dot = current @ weights.T
        # Floor quotient + remainder implements nearest division, with negative
        # and positive ties toward +infinity, without overflowing 2 * dot.
        current = dot // sf + (dot % sf >= (sf + 1) // 2) + bias
        maximum = max(maximum, _magnitude(current))
        by_row = np.maximum(by_row, np.max(np.abs(current.astype(object)), axis=1))
        if layer["params"] == [1]:
            current = np.maximum(current, 0)
        elif layer["params"] != [0]:
            raise ValueError("Only no activation or fused ReLU is supported")
        current_id = outputs[0]
        used_ids.add(current_id)
    final = layers[-1]
    if (
        final["inp_idxes"] != [current_id]
        or len(final["out_idxes"]) != 1
        or final["out_idxes"][0] in used_ids
        or current.shape[1] != 1
        or not model["out_idxes"]
        or model["out_idxes"][-1] != final["out_idxes"][0]
    ):
        raise ValueError(
            "Expected a terminal scalar Logistic matching the public score"
        )
    logits = current.reshape(-1)
    scaled = logits.astype(np.float64) / sf
    # Stable sigmoid for extreme negative logits; matches ordinary sigmoid
    # without exponential overflow in the reference.
    probability = np.exp(-np.logaddexp(0.0, -scaled))
    scores = np.floor(probability * sf + 0.5).astype(np.int64)
    return Inference(scores, logits, maximum, by_row)
