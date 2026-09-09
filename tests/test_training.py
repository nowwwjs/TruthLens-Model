from truthlens.training import EarlyStopping


def test_early_stopping_resets_after_improvement():
    stopper = EarlyStopping(patience=2, min_delta=0.01, mode="max")
    assert not stopper.update(0.5)
    assert not stopper.update(0.505)
    assert not stopper.update(0.52)
    assert not stopper.update(0.515)
    assert stopper.update(0.51)
