import copy
import itertools
import msgpack
import numpy as np
import pytest

from scripts.circuit_plan import CircuitPlan
from scripts.fixed_point import evaluate
from scripts.zk import reference, write_model


def graph():
    # At scale 2, first FC computes [a-b, -a+b+1], then ReLU.
    # Second FC computes 2*h0-h1; third adds 1 before sigmoid.
    return {"global_sf": 2, "inp_idxes": [0], "out_idxes": [1, 2, 4, 5, 7, 8, 11],
            "tensors": [
                {"idx": 1, "shape": [2, 2], "data": [2, -2, -2, 2]},
                {"idx": 2, "shape": [2], "data": [0, 2]},
                {"idx": 4, "shape": [1, 2], "data": [4, -2]},
                {"idx": 5, "shape": [1], "data": [0]},
                {"idx": 7, "shape": [1, 1], "data": [2]},
                {"idx": 8, "shape": [1], "data": [2]}],
            "layers": [
                {"layer_type": "FullyConnected", "inp_idxes": [0, 1, 2], "out_idxes": [3], "params": [1], "implementation": 0},
                {"layer_type": "FullyConnected", "inp_idxes": [3, 4, 5], "out_idxes": [6], "params": [0], "implementation": 2},
                {"layer_type": "FullyConnected", "inp_idxes": [6, 7, 8], "out_idxes": [9], "params": [0], "implementation": 5},
                {"layer_type": "Logistic", "inp_idxes": [9], "out_idxes": [11], "params": [], "implementation": 0}]}


def test_three_layers_follow_tensor_ids_relu_bias_and_public_score():
    result = evaluate(graph(), np.array([[1, 0], [0, 1], [-1, 0]]))
    assert result.logits.tolist() == [6, -2, -2]
    assert result.scores.tolist() == [2, 1, 1]
    assert result.max_abs_intermediate == 6
    scores, logits = reference(graph(), [[1, 0]])
    assert scores.tolist() == [2] and logits.tolist() == [6]


def test_range_summary_includes_hidden_values_before_relu():
    model = graph()
    model["tensors"][1]["data"] = [-40, 2]
    result = evaluate(model, [[1, 0]])
    assert result.max_abs_intermediate == 38
    assert result.logits.tolist() == [2]


@pytest.mark.parametrize("failure", ["activation", "branch", "score", "vector", "overflow", "constants"])
def test_unsupported_or_unsafe_graph_is_rejected(failure):
    model = graph()
    if failure == "activation": model["layers"][0]["params"] = [3]
    if failure == "branch": model["layers"][1]["inp_idxes"][0] = 0
    if failure == "score": model["out_idxes"][-1] = 9
    if failure == "vector": model["layers"] = model["layers"][:1] + [{"layer_type": "Logistic", "inp_idxes": [3], "out_idxes": [11]}]
    if failure == "overflow": model["tensors"][0]["data"] = [2**62, 0, 0, 2**62]
    if failure == "constants": model["tensors"][0]["data"] = [0.5, 0, 0, 0.5]
    with pytest.raises(ValueError): evaluate(model, [[1, 0]])


@pytest.mark.parametrize("x", [[[float("nan"), 0]], [[float("inf"), 0]], [[1e30, 0]], [], [1, 0]])
def test_invalid_inputs_fail_before_integer_conversion(x):
    with pytest.raises(ValueError): evaluate(graph(), x)


def test_every_three_layer_plan_has_a_distinct_identity():
    plans = [CircuitPlan(p) for p in itertools.product(range(6), repeat=3)]
    assert len({p.key for p in plans}) == 216
    assert CircuitPlan((0, 2, 5)).key == "fc-v1-0-2-5"
    assert CircuitPlan.from_model(graph()) == CircuitPlan((0, 2, 5))


@pytest.mark.parametrize("value", [(), (6,), (-1,), (True,), (1.0,), [1]])
def test_invalid_implementation_identity_is_rejected(value):
    with pytest.raises(ValueError): CircuitPlan(value)


def test_writer_preserves_or_explicitly_assigns_complete_plans(tmp_path):
    model = graph()
    before = copy.deepcopy(model)
    path = tmp_path / "model.msgpack"
    preserved = write_model(model, path, implementation=None)
    assert CircuitPlan.from_model(preserved) == CircuitPlan((0, 2, 5))
    variant = write_model(model, path, implementation=CircuitPlan((5, 3, 1)))
    assert CircuitPlan.from_model(variant) == CircuitPlan((5, 3, 1))
    assert msgpack.unpackb(path.read_bytes()) == variant
    assert variant["tensors"] == model["tensors"]
    assert model == before
    scalar = write_model(model, path, implementation=2)
    assert CircuitPlan.from_model(scalar) == CircuitPlan((2, 2, 2))
    saved = path.read_bytes()
    with pytest.raises(ValueError): write_model(model, path, implementation=CircuitPlan((0,)))
    assert path.read_bytes() == saved
