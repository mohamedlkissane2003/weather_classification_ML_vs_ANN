import numpy as np
import pytest


def test_ann_trains_and_predicts_when_tensorflow_is_available():
    pytest.importorskip("tensorflow")
    from weather_classification.models import train_ann

    x_train = np.array([[-2.0, 0.0], [-1.5, 0.2], [0.0, 2.0], [0.2, 1.5], [2.0, 0.0], [1.5, 0.2]])
    y_train = np.array([0, 0, 1, 1, 2, 2])
    model, history = train_ann(x_train, y_train, x_train, y_train, epochs=1, batch_size=2)
    assert model.predict(x_train, verbose=0).shape == (6, 3)
    assert "loss" in history.history