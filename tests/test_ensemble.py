import pytest

from truthlens.ensemble import validate_ensemble_weights, weighted_soft_vote


def test_weighted_soft_vote_scalar_and_batch():
    weights = {"efficientnet": 0.8, "mobilenet": 0.2}
    assert weighted_soft_vote({"efficientnet": 0.75, "mobilenet": 0.25}, weights) == pytest.approx(0.65)
    assert weighted_soft_vote(
        {"efficientnet": [0.75, 0.2], "mobilenet": [0.25, 0.8]}, weights
    ) == pytest.approx([0.65, 0.32])


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError):
        validate_ensemble_weights({"efficientnet": 0.8, "mobilenet": 0.3})
