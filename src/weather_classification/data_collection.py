"""Open-Meteo HTTP client and response validation."""

import logging
from typing import Mapping

import pandas as pd
import requests

from .config import CITIES, RAW_COLUMNS, Settings

LOGGER = logging.getLogger(__name__)
HOURLY_VARIABLES = ("temperature_2m", "relativehumidity_2m", "windspeed_10m", "surface_pressure", "cloudcover", "weathercode")


class DataCollectionError(RuntimeError):
    """Raised when an API response cannot produce valid observations."""


def _city_frame(city: str, payload: Mapping[str, object]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping) or not all(key in hourly for key in ("time", *HOURLY_VARIABLES)):
        raise DataCollectionError(f"Open-Meteo response for {city} is missing hourly variables")
    lengths = {len(hourly[key]) for key in ("time", *HOURLY_VARIABLES)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise DataCollectionError(f"Open-Meteo response for {city} has inconsistent or empty hourly data")
    return pd.DataFrame({
        "city": city,
        "datetime": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relativehumidity_2m"],
        "wind_speed": hourly["windspeed_10m"],
        "surface_pressure": hourly["surface_pressure"],
        "cloud_cover": hourly["cloudcover"],
        "weathercode": hourly["weathercode"],
    }, columns=RAW_COLUMNS)


def collect_weather_data(settings: Settings, cities: Mapping[str, tuple[float, float]] = CITIES) -> pd.DataFrame:
    """Collect hourly observations for each configured city.

    The forecast endpoint is requested with ``past_days``; therefore the dataset
    contains recent historical observations plus the current forecast boundary,
    rather than an unlimited historical record.
    """
    if not cities:
        raise DataCollectionError("No cities are configured for collection")
    frames = []
    with requests.Session() as session:
        for city, (latitude, longitude) in cities.items():
            params = {
                "latitude": latitude, "longitude": longitude,
                "hourly": ",".join(HOURLY_VARIABLES), "past_days": settings.past_days,
                "forecast_days": 1, "timezone": "UTC",
            }
            try:
                response = session.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=settings.timeout_seconds)
                response.raise_for_status()
                frames.append(_city_frame(city, response.json()))
            except (requests.RequestException, ValueError, TypeError, DataCollectionError) as exc:
                raise DataCollectionError(f"Could not collect data for {city}: {exc}") from exc
    if not frames:
        raise DataCollectionError("Open-Meteo returned no city observations")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["city", "datetime"])
    LOGGER.info("Collected %d observations for %d cities", len(result), len(frames))
    return result