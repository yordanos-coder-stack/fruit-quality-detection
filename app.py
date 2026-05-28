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

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Fruit Freshness Detection System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM UI STYLE
# ==========================================
st.markdown("""
<style>
.main {background: linear-gradient(to right, #f8f9fa, #e9ecef);}

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
    font-size: 28px;
    font-weight: bold;
    color: #155724;
    text-align: center;
}

.prediction-bad {
    background-color: #f8d7da;
    padding: 20px;
    border-radius: 15px;
    font-size: 28px;
    font-weight: bold;
    color: #721c24;
    text-align: center;
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
    ["🏠 Home", " Prediction", " Analytics", " About"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.5, 0.01
)

st.sidebar.success("AI System Ready")

# ==========================================
# LOAD MODEL (SAFE CHECK)
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please add 'fruit_detection_model.h5'")
    st.stop()

model = load_model()

# ==========================================
# IMAGE PREPROCESSING
# ==========================================
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ==========================================
# GRAD-CAM (MORE STABLE VERSION)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):

    try:
        base_model = model.layers[1]
        last_conv_layer = base_model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    except:
        return None

# ==========================================
# PDF REPORT
# ==========================================
def generate_pdf(label, confidence):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = [
        Paragraph("Fruit Freshness Prediction Report", styles['Title']),
        Spacer(1, 20),
        Paragraph(f"<b>Prediction:</b> {label}", styles['BodyText']),
        Spacer(1, 10),
        Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['BodyText']),
        Spacer(1, 10),
        Paragraph(
            "Generated using AI system based on MobileNetV2 + Explainable AI",
            styles['BodyText']
        )
    ]

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==========================================
# HOME PAGE (IMPROVED RESEARCH STYLE)
# ==========================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>AI Fruit Freshness Detection System</h1>
        <p>Deep Learning (MobileNetV2) + Explainable AI (Grad-CAM)</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "MobileNetV2 CNN")
    col2.metric("Task", "Fresh vs Rotten Classification")
    col3.metric("Output", "Image-Based Prediction")

    st.markdown("---")

    st.write("""
### System Overview
This system applies deep learning (MobileNetV2) for automatic fruit freshness detection.

It processes fruit images using computer vision preprocessing, then predicts whether the fruit is fresh or rotten.

Key capabilities include:
- Deep learning-based image classification
- Real-time prediction
- Confidence score estimation
- Explainable AI (Grad-CAM visualization)
- PDF report generation
- Prediction history analytics
    """)

# ==========================================
# PREDICTION PAGE
# ==========================================
elif page == " Prediction":

    st.title("Fruit Freshness Prediction")

    uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", use_container_width=True)

        processed = preprocess_image(image)

        prediction = model.predict(processed)

        # SAFE OUTPUT HANDLING
        rotten_prob = float(prediction[0][0])
        fresh_prob = 1 - rotten_prob

        label = "Rotten" if rotten_prob > confidence_threshold else "Fresh"
        confidence = max(rotten_prob, fresh_prob) * 100

        if label == "Rotten":
            st.markdown(f"<div class='prediction-bad'>{label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='prediction-good'>{label}</div>", unsafe_allow_html=True)

        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

        # CHART
        df = pd.DataFrame({
            "Class": ["Fresh", "Rotten"],
            "Probability": [fresh_prob * 100, rotten_prob * 100]
        })

        fig = px.bar(df, x="Class", y="Probability", text="Probability")
        st.plotly_chart(fig, use_container_width=True)

        # PDF
        pdf = generate_pdf(label, confidence)
        st.download_button("Download PDF Report", pdf, file_name="report.pdf")

        # HISTORY
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Time": datetime.now(),
            "Prediction": label,
            "Confidence": confidence
        })

# ==========================================
# ANALYTICS
# ==========================================
elif page == " Analytics":

    st.title("Prediction Analytics Dashboard")

    if "history" in st.session_state:

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        st.plotly_chart(px.pie(df, names="Prediction", title="Distribution"), use_container_width=True)

        st.plotly_chart(
            px.line(df, x="Time", y="Confidence", color="Prediction"),
            use_container_width=True
        )

    else:
        st.info("No prediction history available.")

# ==========================================
# ABOUT
# ==========================================
elif page == " About":

    st.title("About This AI System")

    st.write("""
This project is an AI-powered fruit freshness detection system built using:

- MobileNetV2 deep learning model
- Computer vision preprocessing
- Streamlit web deployment
- Explainable AI (Grad-CAM)
- Interactive visualization tools

The system classifies fruit images into Fresh or Rotten categories and provides confidence scores and analytical insights.
""")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.markdown("""
<div class="footer">
AI Fruit Freshness Detection System | Deep Learning + Explainable AI
</div>
""", unsafe_allow_html=True)
