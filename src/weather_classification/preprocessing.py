"""Validation, cleaning, target labeling, and leakage-safe splitting."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, RAW_COLUMNS, WEATHER_CODE_TO_CLASS


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot support the experiment."""


def classify_weather(code: int | float) -> int:
    """Map a supported WMO code to the project's single target definition."""
    if pd.isna(code):
        raise ValueError("Weather code cannot be missing")
    numeric_code = float(code)
    if not numeric_code.is_integer():
        raise ValueError(f"Unsupported non-integer WMO weather code: {code}")
    try:
        return WEATHER_CODE_TO_CLASS[int(numeric_code)]
    except KeyError as exc:
        raise ValueError(f"Unsupported WMO weather code: {int(numeric_code)}") from exc


def clean_data(frame: pd.DataFrame, remove_iqr_outliers: bool = True) -> pd.DataFrame:
    """Clean observations using physical bounds and optional per-city IQR filtering.

    Missing feature rows are removed, never filled across unrelated city time
    series. Statistical filtering is per city and only used for groups with
    enough observations; valid extreme weather remains within physical bounds.
    """
    required = set(RAW_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise DatasetValidationError(f"Missing columns: {sorted(missing)}")
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce", utc=True)
    numeric = ["temperature", "humidity", "wind_speed", "surface_pressure", "cloud_cover", "weathercode"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["city", "datetime", *numeric]).drop_duplicates(subset=["city", "datetime"])
    result = result[
        result["humidity"].between(0, 100)
        & result["temperature"].between(-30, 55)
        & (result["wind_speed"] >= 0)
        & result["cloud_cover"].between(0, 100)
        & result["surface_pressure"].between(800, 1100)
    ]
    if remove_iqr_outliers and len(result) >= 8:
        measurements = ["temperature", "humidity", "wind_speed", "surface_pressure", "cloud_cover"]
        def keep_group(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) < 8:
                return group
            quantiles = group[measurements].quantile([0.25, 0.75])
            q1, q3 = quantiles.loc[0.25], quantiles.loc[0.75]
            bounds = (q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))
            return group[((group[measurements] >= bounds[0]) & (group[measurements] <= bounds[1])).all(axis=1)]
        groups = [keep_group(group) for _, group in result.groupby("city", sort=False)]
        result = pd.concat(groups, ignore_index=True) if groups else result.iloc[0:0]
    result["target"] = result["weathercode"].map(classify_weather)
    return result.sort_values(["datetime", "city"]).reset_index(drop=True)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the hour and month features used by both models."""
    result = frame.copy()
    datetime = pd.to_datetime(result["datetime"], utc=True)
    result["hour"] = datetime.dt.hour
    result["month"] = datetime.dt.month
    return result


def class_distribution(frame: pd.DataFrame) -> dict[int, int]:
    """Return counts for every target class, including classes with zero rows."""
    return {label: int((frame["target"] == label).sum()) for label in sorted({0, 1, 2})}


def validate_dataset(frame: pd.DataFrame, minimum_rows: int = 30, minimum_class_examples: int = 2) -> None:
    """Validate row count and sufficient examples for each target class."""
    if len(frame) < minimum_rows:
        raise DatasetValidationError(f"Need at least {minimum_rows} observations, found {len(frame)}")
    if "target" not in frame:
        raise DatasetValidationError("Dataset is missing the target column")
    distribution = class_distribution(frame)
    insufficient = {label: count for label, count in distribution.items() if count < minimum_class_examples}
    if insufficient:
        raise DatasetValidationError(f"Each class needs at least {minimum_class_examples} examples; counts are {distribution}")


def chronological_split(frame: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at one common timestamp boundary for every city."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    ordered = frame.sort_values(["datetime", "city"]).reset_index(drop=True)
    timestamps = ordered["datetime"].drop_duplicates().sort_values().reset_index(drop=True)
    cut = max(1, int(len(timestamps) * (1 - test_size)))
    cutoff = timestamps.iloc[cut - 1]
    train = ordered[ordered["datetime"] <= cutoff].copy()
    test = ordered[ordered["datetime"] > cutoff].copy()
    if train.empty or test.empty:
        raise DatasetValidationError("Chronological split produced an empty partition")
    return train, test


def fit_transform_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit the scaler on training data only and transform both partitions."""
    scaler = StandardScaler()
    return scaler.fit_transform(train[FEATURES]), scaler.transform(test[FEATURES]), scaler