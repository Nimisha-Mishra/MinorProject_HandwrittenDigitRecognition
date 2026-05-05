"""
Flask web application for Handwritten Digit Recognition.

Before running, train the model first:
    python train.py

Then start the app:
    python app.py

Open http://127.0.0.1:5000 in a browser, draw a digit on the canvas,
and click "Predict" to see the result.
"""

import base64
import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_model.h5")
model = None


def load_model() -> None:
    """Load the saved Keras model into the global `model` variable."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please run 'python train.py' first."
        )
    model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_data: str) -> np.ndarray:
    """
    Decode a base64 PNG image from the browser canvas, convert it to
    a 28×28 grayscale array suitable for the MNIST model, and return
    it as a (1, 28, 28, 1) float32 array.
    """
    # Strip the data-URL prefix ("data:image/png;base64,…")
    if "," in image_data:
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Composite onto a white background so transparent areas become white
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.paste(image, mask=image.split()[3])
    image = background.convert("L")  # grayscale

    # MNIST uses white digits on a black background; invert if needed
    image = ImageOps.invert(image)

    image = image.resize((28, 28), Image.LANCZOS)

    img_array = np.array(image, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 28, 28, 1)
    return img_array


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    image_data = data.get("image", "")

    if not image_data:
        return jsonify({"error": "No image data received."}), 400

    try:
        img_array = preprocess_image(image_data)
        predictions = model.predict(img_array, verbose=0)
        digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        # Build a list of (digit, probability) pairs for all 10 classes
        probabilities = [
            {"digit": i, "probability": round(float(predictions[0][i]) * 100, 2)}
            for i in range(10)
        ]

        return jsonify(
            {
                "digit": digit,
                "confidence": round(confidence, 2),
                "probabilities": probabilities,
            }
        )
    except Exception:  # pragma: no cover
        return jsonify({"error": "Prediction failed. Please try again."}), 500


if __name__ == "__main__":
    load_model()
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
