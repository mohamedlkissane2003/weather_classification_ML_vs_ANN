"""Non-interactive report figures."""

from pathlib import Path


def save_eda_figures(frame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.histplot(frame["temperature"], kde=True)
    plt.title("Temperature distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_distribution.png", dpi=150)
    plt.close()
    numeric = ["temperature", "humidity", "wind_speed", "surface_pressure", "cloud_cover", "hour", "month"]
    sns.heatmap(frame[numeric].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()


def save_confusion_matrix(matrix, labels, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_history(history, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if history is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, title in (("loss", "ANN loss"), ("accuracy", "ANN accuracy")):
        plt.plot(history.history.get(metric, []), label=f"Train {metric}")
        plt.plot(history.history.get(f"val_{metric}", []), label=f"Validation {metric}")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"ann_{metric}.png", dpi=150)
        plt.close()