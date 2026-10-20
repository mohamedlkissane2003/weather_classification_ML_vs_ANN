import pandas as pd
import pytest

from weather_classification.preprocessing import (
    DatasetValidationError,
    add_features,
    classify_weather,
    clean_data,
    chronological_split,
    fit_transform_features,
)


def make_frame(rows=36):
    return pd.DataFrame({
        "city": ["Rabat"] * rows,
        "datetime": pd.date_range("2025-01-01", periods=rows, freq="h"),
        "temperature": [20 + i % 3 for i in range(rows)],
        "humidity": [50 + i % 10 for i in range(rows)],
        "wind_speed": [5.0] * rows,
        "surface_pressure": [1010.0] * rows,
        "cloud_cover": [20.0] * rows,
        "weathercode": [0, 45, 61] * (rows // 3),
    })


def test_weather_code_mapping_rejects_unknown_codes():
    assert classify_weather(0) == 0
    assert classify_weather(1) == 0
    assert classify_weather(2) == 1
    assert classify_weather(48) == 1
    assert classify_weather(61) == 2
    assert classify_weather(99) == 2
    with pytest.raises(ValueError):
        classify_weather(4)


def test_cleaning_drops_invalid_and_missing_rows_without_filling_across_series():
    frame = make_frame()
    frame.loc[0, "humidity"] = None
    frame.loc[1, "temperature"] = 100
    cleaned = clean_data(frame, remove_iqr_outliers=False)
    assert len(cleaned) == len(frame) - 2
    assert cleaned["humidity"].notna().all()


def test_feature_engineering_and_chronological_split():
    frame = pd.concat([make_frame(), make_frame().assign(city="Marrakech")], ignore_index=True)
    prepared = add_features(clean_data(frame, remove_iqr_outliers=False))
    train, test = chronological_split(prepared, test_size=0.2)
    assert {"hour", "month"}.issubset(prepared.columns)
    assert train["datetime"].max() < test["datetime"].min()
    assert set(train["datetime"]).isdisjoint(test["datetime"])


def test_scaler_is_fitted_on_training_partition_only():
    prepared = add_features(clean_data(make_frame(), remove_iqr_outliers=False))
    train, test = chronological_split(prepared)
    x_train, x_test, scaler = fit_transform_features(train, test)
    assert x_train.shape[1] == 7
    assert scaler.mean_[0] == pytest.approx(train["temperature"].mean())
    assert x_test[:, 0].mean() != pytest.approx(0.0)


def test_validation_requires_three_classes():
    prepared = add_features(clean_data(make_frame(), remove_iqr_outliers=False))
    prepared.loc[:, "target"] = 0
    with pytest.raises(DatasetValidationError):
        from weather_classification.preprocessing import validate_dataset
        validate_dataset(prepared)