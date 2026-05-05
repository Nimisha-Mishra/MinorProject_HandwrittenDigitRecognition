"""
Train a Convolutional Neural Network on the MNIST dataset.

Run this script once to generate the saved model file (mnist_model.h5)
that the Flask web app (app.py) will use for predictions.

Usage:
    python train.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model() -> keras.Model:
    """Build and return a CNN model for digit classification."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )
    return model


def main() -> None:
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension (grayscale → shape (N, 28, 28, 1))
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = build_model()
    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    batch_size = 128
    epochs = 15

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss:     {score[0]:.4f}")
    print(f"Test accuracy: {score[1]*100:.2f}%")

    model_path = os.path.join(os.path.dirname(__file__), "mnist_model.h5")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
