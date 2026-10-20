"""Model factories and training helpers."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight


def class_weights(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def train_logistic_regression(x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 500) -> LogisticRegression:
    model = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    return model


def build_ann(input_features: int, learning_rate: float = 0.001):
    """Build a lightweight feed-forward Keras model."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to train the ANN") from exc
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_features,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_ann(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    random_state: int = 42,
):
    """Train the ANN with an explicit validation period from training data."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(random_state)
    model = build_ann(x_train.shape[1])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_validation, y_validation),
        shuffle=False,
        class_weight=class_weights(y_train),
        callbacks=callbacks,
        verbose=0,
    )
    return model, history