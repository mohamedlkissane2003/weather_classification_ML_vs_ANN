"""Metrics and report serialization."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_predictions(y_true, y_pred, class_names: dict[int, str]) -> dict:
    labels = sorted(class_names)
    report = classification_report(y_true, y_pred, labels=labels, target_names=[class_names[i] for i in labels], output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "per_class": {name: {metric: float(value) for metric, value in values.items() if metric in {"precision", "recall", "f1-score", "support"}} for name, values in report.items() if name in class_names.values()},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_metrics(metrics_by_model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics_by_model, indent=2), encoding="utf-8")


def comparison_frame(metrics_by_model: dict) -> pd.DataFrame:
    return pd.DataFrame([{ "model": name, **{key: values[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")} } for name, values in metrics_by_model.items()])