import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# 1. Page Configuration
st.set_page_config(page_title="Digit Recognizer", layout="wide")
st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a single digit (0-9) in the box below and the CNN will guess it!")

# 2. Load the trained model (Ensure train_model.py has been run first)
@st.cache_resource # This caches the model so it doesn't reload on every drawing stroke
def load_trained_model():
    return load_model('mnist_digit_model.h5')

model = load_trained_model()

# 3. Create two columns (Left for drawing, Right for results)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Draw Here")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="#000000",  # Fixed fill color with some opacity
        stroke_width=20,       # Thicker stroke works better for resizing
        stroke_color="#FFFFFF",# Draw in white
        background_color="#000000", # Black background (like MNIST data)
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("### Prediction")
    if canvas_result.image_data is not None:
        # 4. Image Processing
        # The canvas returns an RGBA image array. We need to convert it to Grayscale 28x28
        img = canvas_result.image_data
        
        # Convert to 8-bit unsigned integer
        img = img.astype('uint8')
        
        # Convert RGBA to Grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        # Resize to 28x28 (the format the model expects)
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values (0 to 1)
        img_normalized = img_resized / 255.0
        
        # Reshape to match the model's input shape (1 image, 28x28, 1 channel)
        img_reshaped = img_normalized.reshape(1, 28, 28, 1)

        # 5. Prediction
        prediction = model.predict(img_reshaped)
        guessed_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display results
        st.write(f"## I think this is a: **{guessed_digit}**")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")
        
        # Optional: Show the panel exactly what the neural network "sees"
        st.write("What the neural network sees (28x28):")
        st.image(img_resized, width=150)