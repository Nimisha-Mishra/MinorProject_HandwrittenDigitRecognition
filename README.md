# Handwritten Digit Recognition

A web application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

---

## Features

- **CNN model** – Two convolutional blocks + dense output layer, achieving ~99% accuracy on the MNIST test set.
- **Interactive canvas** – Draw any digit with mouse or touch in the browser.
- **Instant prediction** – The Flask back-end returns the predicted digit, confidence score, and per-class probability bars.

---

## Project Structure

```
├── train.py          # Train the CNN and save mnist_model.h5
├── app.py            # Flask web application (loads the saved model)
├── requirements.txt  # Python dependencies
├── templates/
│   └── index.html    # Drawing canvas UI
└── static/
    ├── style.css     # Styles
    └── script.js     # Canvas drawing + AJAX prediction
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This downloads MNIST automatically, trains for 15 epochs, and saves `mnist_model.h5` in the project root.

### 3. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser, draw a digit on the canvas, and click **Predict**.

---

## Model Architecture

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | – |
| Dropout | 0.5 |
| Dense (output) | 10 units, Softmax |

**Optimizer:** Adam  **Loss:** Categorical cross-entropy  **Epochs:** 15

---

## Technologies

- Python 3.8+
- TensorFlow / Keras
- Flask
- HTML5 Canvas API
- Pillow (image preprocessing)

