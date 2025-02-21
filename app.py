import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Google Drive file ID
GDRIVE_MODEL_ID = "1os3m_b2PYcvCz33Ku_vzRjSkhBh8Y7e5"
MODEL_PATH = "model.h5"

# Function to download model from Google Drive
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)

    st.success("Model loaded successfully!")
    return load_model(MODEL_PATH, compile=False)

# Load the model
model = download_and_load_model()

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to model input size
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Streamlit UI
st.title("Image Classification App")
st.write("Upload an image, and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image_display)
    prediction = model.predict(img_array)

    # Show prediction
    st.subheader("Prediction:")
    st.write(f"Class: {np.argmax(prediction)}")
    st.write(f"Confidence: {np.max(prediction):.2f}")
