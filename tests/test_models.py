import numpy as np

from weather_classification.models import class_weights, train_logistic_regression


def test_logistic_regression_trains_and_predicts_three_classes():
    x_train = np.array([
        [-2.0, 0.0], [-1.5, 0.2],
        [0.0, 2.0], [0.2, 1.5],
        [2.0, 0.0], [1.5, 0.2],
    ])
    y_train = np.array([0, 0, 1, 1, 2, 2])
    model = train_logistic_regression(x_train, y_train)
    predictions = model.predict(x_train)
    assert set(predictions) == {0, 1, 2}
    assert set(class_weights(y_train)) == {0, 1, 2}