import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Google Drive file ID
GDRIVE_MODEL_ID = "1MGBH4qECimwgJGXLuEv2Y_ZEUV9b0Yql"
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

# Air Quality Classification Information
air_quality_info = {
    "e_Very_Unhealthy": {"Range": "201-300", "Label": "Very Unhealthy", "Description": "Health warnings of emergency conditions. The entire population is more likely to be affected."},
    "f_Severe": {"Range": "301-500", "Label": "Hazardous", "Description": "Serious health effects; everyone should avoid outdoor activities."}
}

# Streamlit UI
st.title("🌍 Air Quality Classification")
st.write("Upload an image to classify the air quality level.")

# File uploader
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image_display)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Show prediction
    st.subheader("Prediction:")
    st.write(f"Class: {predicted_class}")

    # Display air quality information based on the predicted class
    if predicted_class in air_quality_info:
        st.subheader("Air Quality Information:")
        st.write(f"**Range:** {air_quality_info[predicted_class]['Range']}")
        st.write(f"**Label:** {air_quality_info[predicted_class]['Label']}")
        st.write(f"**Description:** {air_quality_info[predicted_class]['Description']}")
    else:
        st.write("No air quality information available for this class.")
