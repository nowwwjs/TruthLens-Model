from truthlens.evaluation import binary_metrics, select_threshold_on_validation


def test_metrics_include_confusion_matrix():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == 0.5


def test_threshold_is_selected_from_passed_validation_values():
    threshold, metrics = select_threshold_on_validation(
        [0, 0, 1, 1],
        [0.1, 0.3, 0.4, 0.9],
        candidates=[0.3, 0.4, 0.5],
    )
    assert threshold == 0.4
    assert metrics["f1"] == 1.0
