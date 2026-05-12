import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = tf.keras.models.load_model("plant_disease_model_keras")

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

st.set_page_config(page_title="Plant Disease Detection")

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((160, 160))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    confidence = np.max(predictions)

    st.success(f"🦠 Disease: {class_names[pred_index]}")
    st.info(f"📊 Confidence: {confidence:.2%}")
