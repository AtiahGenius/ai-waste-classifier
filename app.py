
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #2b5876;
    text-align: center;
    padding-top: 10px;
}

.desc {
    font-size: 17px;
    text-align: center;
    color: #333;
    margin-bottom: 20px;
}

.upload-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.prediction-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #e0f7e9;
    text-align: center;
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
    color: #777;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# TITLE + DESCRIPTION
# -----------------------------------------------------------
st.markdown('<div class="main-title">♻️ AI Waste Classifier</div>', unsafe_allow_html=True)
st.markdown('<p class="desc">Upload an image or take a picture, and the AI will classify it as <b>Recyclable</b>, <b>Biodegradable</b>, or <b>Non-Recyclable</b>.</p>', unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "waste_model_efficientnet.keras")
model = tf.keras.models.load_model(r'C:\Users\prosp\ai-waste-classifier\waste_model_efficientnet.keras')

# -----------------------------------------------------------
# CLASS NAMES + ADVICE
# -----------------------------------------------------------
class_names = ["biodegradable", "non_recyclable", "recyclable"]

ADVICE = {
    "recyclable": "♻️ This item is recyclable. Rinse if necessary and place it in the recycling bin.",
    "biodegradable": "🌱 This item is biodegradable. Compost it or dispose of it with organic waste.",
    "non_recyclable": "🚯 This item is non-recyclable. Dispose of it properly. Avoid burning or littering."
}

CLASS_COLORS = {
    "recyclable": "#4caf50",
    "biodegradable": "#8bc34a",
    "non_recyclable": "#f44336"
}

ICONS = {
    "recyclable": "♻️",
    "biodegradable": "🌱",
    "non_recyclable": "🚯"
}

# -----------------------------------------------------------
# IMAGE PREPROCESSING
# -----------------------------------------------------------
def preprocess_image(image, img_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(img_size)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# -----------------------------------------------------------
# FILE UPLOAD + CAMERA INPUT
# -----------------------------------------------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Take Picture"])

uploaded_file = None

with tab1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

with tab2:
    camera_file = st.camera_input("Take a picture")

    if camera_file is not None:
        uploaded_file = camera_file

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# DISPLAY + CLASSIFY
# -----------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("🔍 Classify Image", use_container_width=True):
        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        probs = tf.nn.softmax(preds[0]).numpy()
        idx = int(np.argmax(probs))

        predicted_class = class_names[idx]
        confidence = float(probs[idx])

        # BEAUTIFUL PREDICTION CARD
        st.markdown(
            f"""
            <div class="prediction-card" style="background-color:{CLASS_COLORS[predicted_class]}20;">
                <h2 style="color:{CLASS_COLORS[predicted_class]}; font-size:32px;">
                    {ICONS[predicted_class]} {predicted_class.replace('_',' ').title()}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence progress bar
        st.write("### Confidence Level")
        st.progress(confidence)

        # Advice
        st.info(ADVICE[predicted_class])

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown('<p class="footer">Developed with using Streamlit & TensorFlow By Geniuz Atiah </p>', unsafe_allow_html=True)