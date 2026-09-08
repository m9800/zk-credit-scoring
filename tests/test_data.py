import numpy as np
import pandas as pd
from scripts.train import fit_preprocessing, transform, split_masks


def test_future_extremes_do_not_change_fitted_preprocessing():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    state = fit_preprocessing(train, ["x"], [])
    before = dict(state)
    values = transform(pd.DataFrame({"x": [1e12, np.nan]}), state)
    assert state == before
    assert np.isfinite(values).all()
    assert values[0, 0] == transform(pd.DataFrame({"x": [state["upper"][0]]}), state)[0, 0]


def test_chronological_gap_and_disjoint_splits():
    timestamps = np.arange(1000) * 86400.0
    masks, _ = split_masks(timestamps, 30)
    assert np.max(sum(m.astype(int) for m in masks.values())) == 1
    assert timestamps[masks["validation"]].min() - timestamps[masks["train"]].max() > 30 * 86400
    assert timestamps[masks["test"]].min() - timestamps[masks["validation"]].max() > 30 * 86400
