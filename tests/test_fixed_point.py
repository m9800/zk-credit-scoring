import numpy as np
from scripts.zk import reference


def test_fixed_point_rounding_and_sigmoid_boundary():
    model = {"global_sf": 2, "tensors": [
        {"idx": 1, "shape": [1, 1], "data": [1]},
        {"idx": 2, "shape": [1], "data": [0]}],
        "layers": [{"inp_idxes": [0, 1, 2]}]}
    _, logits = reference(model, np.array([[-0.5], [0.0], [0.5]], dtype=np.float32))
    assert logits.tolist() == [0, 0, 1]
