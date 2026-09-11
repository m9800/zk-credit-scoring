import numpy as np
import json
from pathlib import Path
from scripts.zk import reference


def test_fixed_point_rounding_and_sigmoid_boundary():
    model = {"global_sf": 2, "tensors": [
        {"idx": 1, "shape": [1, 1], "data": [1]},
        {"idx": 2, "shape": [1], "data": [0]}],
        "inp_idxes": [0], "out_idxes": [4],
        "layers": [{"layer_type": "FullyConnected", "inp_idxes": [0, 1, 2],
                    "out_idxes": [3], "params": [0]},
                   {"layer_type": "Logistic", "inp_idxes": [3], "out_idxes": [4]}]}
    _, logits = reference(model, np.array([[-0.5], [0.0], [0.5]], dtype=np.float32))
    assert logits.tolist() == [0, 0, 1]


def test_retained_logistic_model_matches_archived_circuit_scores_and_logits():
    fixture = json.loads((Path(__file__).parent / "fixtures/logistic-reference.json").read_text())
    scores, logits = reference(fixture["model"], [case["input"] for case in fixture["cases"]])
    assert scores.tolist() == [case["score"] for case in fixture["cases"]]
    assert logits.tolist() == [case["logit"] for case in fixture["cases"]]
