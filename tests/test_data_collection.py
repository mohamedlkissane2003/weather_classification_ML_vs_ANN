import pytest

from weather_classification.data_collection import DataCollectionError, _city_frame


def test_api_response_is_validated():
    payload = {"hourly": {"time": ["2025-01-01T00:00"], "temperature_2m": [20], "relativehumidity_2m": [50], "windspeed_10m": [4], "surface_pressure": [1010], "cloudcover": [10], "weathercode": [0]}}
    frame = _city_frame("Rabat", payload)
    assert list(frame.columns) == ["city", "datetime", "temperature", "humidity", "wind_speed", "surface_pressure", "cloud_cover", "weathercode"]
    with pytest.raises(DataCollectionError):
        _city_frame("Rabat", {"hourly": {"time": []}})