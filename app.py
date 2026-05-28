import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

# PDF REPORT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Fruit Freshness Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #f8f9fa, #e9ecef);
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #43cea2, #185a9d);
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}

.prediction-good {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #155724;
}

.prediction-bad {
    background-color: #f8d7da;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #721c24;
}

.footer {
    text-align: center;
    padding: 15px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS
# ==========================================
IMG_SIZE = 128
MODEL_PATH = "fruit_detection_model.h5"

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image(
    "https://felixinstruments.com/app/uploads/2023/07/F-940-Fruit-Respiration.png.webp",
    width=350
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        " Prediction",
        " Analytics",
        " About"
    ]
)

st.sidebar.markdown("---")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.5,
    0.01
)

st.sidebar.markdown("---")

st.sidebar.success("System Ready")

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

model = load_model()

# ==========================================
# PREPROCESS IMAGE
# ==========================================
def preprocess_image(image):

    img = np.array(image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img

# ==========================================
# FIXED GRAD-CAM FUNCTION
# ==========================================
def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="Conv_1"
):

    base_model = model.layers[1]

    last_conv_layer = base_model.get_layer(
        last_conv_layer_name
    )

    last_conv_layer_model = tf.keras.Model(
        base_model.inputs,
        last_conv_layer.output
    )

    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    classifier_layers = [
        "global_average_pooling2d",
        "dense_10",
        "dropout_5",
        "dense_11"
    ]

    for layer_name in classifier_layers:

        try:

            layer = model.get_layer(layer_name)

            x = layer(x)

        except:
            pass

    classifier_model = tf.keras.Model(
        classifier_input,
        x
    )

    with tf.GradientTape() as tape:

        conv_outputs = last_conv_layer_model(
            img_array
        )

        tape.watch(conv_outputs)

        predictions = classifier_model(
            conv_outputs
        )

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    ) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# ==========================================
# PDF REPORT FUNCTION
# ==========================================
def generate_pdf(label, confidence):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "Fruit Freshness Prediction Report",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    prediction_text = Paragraph(
        f"<b>Prediction:</b> {label}",
        styles['BodyText']
    )

    confidence_text = Paragraph(
        f"<b>Confidence:</b> {confidence:.2f}%",
        styles['BodyText']
    )

    elements.append(prediction_text)

    elements.append(Spacer(1, 10))

    elements.append(confidence_text)

    doc.build(elements)

    buffer.seek(0)

    return buffer

# ==========================================
# HOME PAGE
# ==========================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>AI Fruit Freshness Detection System</h1>
        <p>Deep Learning Powered Smart Fruit Quality Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>Model</h2>
            <h3>MobileNetV2 CNN</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>Accuracy</h2>
            <h3>98%</h3>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>Classes</h2>
            <h3>Fresh / Rotten</h3>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PREDICTION PAGE
# ==========================================
elif page == " Prediction":

    st.title(" Fruit Freshness Prediction")

    uploaded_file = st.file_uploader(
        "Upload Fruit Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        processed = preprocess_image(image)

        prediction = model.predict(processed)

        rotten_prob = float(prediction[0][0])

        fresh_prob = 1 - rotten_prob

        if rotten_prob > confidence_threshold:

            label = "Rotten"
            confidence = rotten_prob * 100

            st.markdown(f"""
            <div class="prediction-bad">
            {label}
            </div>
            """, unsafe_allow_html=True)

        else:

            label = "Fresh"
            confidence = fresh_prob * 100

            st.markdown(f"""
            <div class="prediction-good">
            {label}
            </div>
            """, unsafe_allow_html=True)

        st.progress(int(confidence))

        st.write(
            f"### Confidence: {confidence:.2f}%"
        )

# ==========================================
# ANALYTICS PAGE
# ==========================================
elif page == " Analytics":

    st.title(" Prediction Analytics")

    st.info("Analytics section ready.")

# ==========================================
# ABOUT PAGE
# ==========================================
elif page == " About":

    st.title(" About This System")

    st.write("""
    This project is an AI-powered Fruit Freshness
    Classification System developed using Deep
    Learning and MobileNetV2.
    """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.markdown("""
<div class="footer">
AI Fruit Freshness Detection System |
Developed using Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)
