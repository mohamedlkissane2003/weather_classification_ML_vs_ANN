"""Command-line workflow for collection, training, and reporting."""

import argparse
import json
import logging
import pickle

import pandas as pd

from .config import CLASS_NAMES, FEATURES, Settings
from .data_collection import collect_weather_data
from .evaluation import comparison_frame, evaluate_predictions, save_metrics
from .models import train_ann, train_logistic_regression
from .preprocessing import add_features, class_distribution, clean_data, chronological_split, fit_transform_features, validate_dataset
from .visualization import save_confusion_matrix, save_eda_figures, save_history


def run(settings: Settings, input_path: str | None = None, train_ann_model: bool = True) -> dict:
    settings.create_directories()
    raw = pd.read_csv(input_path) if input_path else collect_weather_data(settings)
    raw.to_csv(settings.data_raw / "weather_raw.csv", index=False)
    cleaned = add_features(clean_data(raw))
    validate_dataset(cleaned)
    cleaned.to_csv(settings.data_processed / "weather_model_dataset.csv", index=False)
    save_eda_figures(cleaned, settings.reports_dir / "figures")
    train, test = chronological_split(cleaned, settings.test_size)
    fit, validation = chronological_split(train, settings.validation_size)
    logging.getLogger(__name__).info("Class counts: full=%s train=%s validation=%s test=%s", class_distribution(cleaned), class_distribution(train), class_distribution(validation), class_distribution(test))
    if set(fit["target"].unique()) != {0, 1, 2}:
        raise ValueError(f"Training period does not contain all target classes: {class_distribution(fit)}")
    x_fit, x_validation, scaler = fit_transform_features(fit, validation)
    x_test = scaler.transform(test[FEATURES])
    y_fit, y_validation, y_test = fit["target"].to_numpy(), validation["target"].to_numpy(), test["target"].to_numpy()
    metrics_by_model = {}
    logistic = train_logistic_regression(x_fit, y_fit, settings.logistic_max_iter)
    metrics_by_model["logistic_regression"] = evaluate_predictions(y_test, logistic.predict(x_test), CLASS_NAMES)
    with (settings.models_dir / "logistic_bundle.pkl").open("wb") as handle:
        pickle.dump({"model": logistic, "scaler": scaler, "features": FEATURES}, handle)
    if train_ann_model:
        ann, history = train_ann(x_fit, y_fit, x_validation, y_validation, settings.ann_epochs, settings.ann_batch_size, settings.random_state)
        ann.save(settings.models_dir / "ann.keras")
        metrics_by_model["ann"] = evaluate_predictions(y_test, ann.predict(x_test, verbose=0).argmax(axis=1), CLASS_NAMES)
        save_history(history, settings.reports_dir / "figures")
    for name, values in metrics_by_model.items():
        save_confusion_matrix(values["confusion_matrix"], list(CLASS_NAMES.values()), f"{name} confusion matrix", settings.reports_dir / "figures" / f"{name}_confusion_matrix.png")
    save_metrics(metrics_by_model, settings.reports_dir / "metrics.json")
    (settings.reports_dir / "class_distribution.json").write_text(json.dumps({
        "full": class_distribution(cleaned),
        "train": class_distribution(train),
        "validation": class_distribution(validation),
        "test": class_distribution(test),
        "note": "Small or absent classes in a partition limit the reliability of per-class evaluation.",
    }, indent=2), encoding="utf-8")
    comparison_frame(metrics_by_model).to_csv(settings.reports_dir / "model_comparison.csv", index=False)
    return metrics_by_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Use a local raw CSV instead of calling Open-Meteo")
    parser.add_argument("--no-ann", action="store_true", help="Run only the Logistic Regression baseline")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = run(Settings(), args.input, not args.no_ann)
    print(comparison_frame(results).to_string(index=False))


if __name__ == "__main__":
    main()