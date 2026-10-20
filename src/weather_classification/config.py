"""Central configuration for data collection and experiments."""

from dataclasses import dataclass, field
from pathlib import Path


FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "surface_pressure",
    "cloud_cover",
    "hour",
    "month",
]
RAW_COLUMNS = ["city", "datetime", "temperature", "humidity", "wind_speed", "surface_pressure", "cloud_cover", "weathercode"]
CLASS_NAMES = {0: "Clear", 1: "Cloudy/Fog", 2: "Precipitation/Storm"}
WEATHER_CODE_TO_CLASS = {
    **{code: 0 for code in (0, 1)},
    **{code: 1 for code in (2, 3, 45, 48)},
    **{code: 2 for code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99)},
}
CITIES = {
    "Marrakech": (31.63, -8.00), "Casablanca": (33.59, -7.62),
    "Rabat": (34.01, -6.83), "Fes": (34.04, -5.00),
    "Agadir": (30.42, -9.60), "Tangier": (35.77, -5.80),
    "Oujda": (34.68, -1.91), "Meknes": (33.89, -5.56),
    "Laayoune": (27.15, -13.20),
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings; paths are relative to the repository root."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    past_days: int = 30
    timeout_seconds: float = 20.0
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    logistic_max_iter: int = 500
    ann_epochs: int = 60
    ann_batch_size: int = 32

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def create_directories(self) -> None:
        for path in (self.data_raw, self.data_processed, self.models_dir, self.reports_dir / "figures"):
            path.mkdir(parents=True, exist_ok=True)