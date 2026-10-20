from weather_classification.evaluation import comparison_frame, evaluate_predictions


def test_evaluation_returns_required_metrics():
    result = evaluate_predictions([0, 1, 2, 2], [0, 1, 1, 2], {0: "clear", 1: "cloudy", 2: "precipitation"})
    assert set(("accuracy", "balanced_accuracy", "macro_f1", "classification_report", "per_class", "confusion_matrix")) <= result.keys()
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert set(comparison_frame({"baseline": result}).columns) == {"model", "accuracy", "balanced_accuracy", "macro_f1"}