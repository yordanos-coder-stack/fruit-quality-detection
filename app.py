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
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ==========================================
# GRAD-CAM FUNCTION (UNCHANGED)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    base_model = model.layers[1]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    last_conv_layer_model = tf.keras.Model(
        base_model.inputs,
        last_conv_layer.output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
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

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = last_conv_layer_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# ==========================================
# PDF REPORT
# ==========================================
def generate_pdf(label, confidence):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph(
        "AI Fruit Freshness Classification Report",
        styles['Title']
    ))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        f"<b>Prediction Result:</b> {label}",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        f"<b>Model Confidence:</b> {confidence:.2f}%",
        styles['BodyText']
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==========================================
# HOME PAGE (UPDATED CONTENT ONLY)
# ==========================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>AI-Powered Fruit Freshness Detection System</h1>
        <p>
        A Deep Learning-based Computer Vision system using MobileNetV2 to classify fruits as Fresh or Rotten with real-time prediction and explainable AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>Architecture</h2>
            <h3>MobileNetV2 CNN</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>System Type</h2>
            <h3>Deep Learning Vision</h3>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>Output Classes</h2>
            <h3>Fresh / Rotten</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Features",
        "Workflow",
        "Samples"
    ])

    with tab1:
        st.write("""
        • Real-time fruit freshness classification  
        • MobileNetV2 transfer learning model  
        • Explainable AI using Grad-CAM heatmaps  
        • Confidence score visualization  
        • PDF report generation  
        • Prediction history tracking  
        • Interactive analytics dashboard  
        """)

    with tab2:

        workflow = pd.DataFrame({
            "Step": [
                "Image Upload",
                "Image Preprocessing",
                "Feature Extraction (CNN)",
                "Prediction (Softmax)",
                "Explainability (Grad-CAM)"
            ],
            "Order": [1,2,3,4,5]
        })

        fig = px.line(
            workflow,
            x="Order",
            y="Order",
            text="Step",
            markers=True,
            title="AI System Pipeline"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.image([
            "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce",
            "https://images.unsplash.com/photo-1574226516831-e1dff420e37f"
        ], width=300)

# ==========================================
# PREDICTION PAGE
# ==========================================
elif page == " Prediction":

    st.title("Fruit Freshness Prediction System")

    prediction_method = st.radio(
        "Select Input Method",
        ["Upload Image", "Use Camera"]
    )

    image = None

    if prediction_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload Fruit Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    else:
        camera_image = st.camera_input("Take a Picture")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")

    if image is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

        with col2:

            with st.spinner("Running AI Model..."):

                processed = preprocess_image(image)
                prediction = model.predict(processed)

                rotten_prob = float(prediction[0][0])
                fresh_prob = 1 - rotten_prob

                if rotten_prob > confidence_threshold:
                    label = "Rotten"
                    confidence = rotten_prob * 100
                    st.markdown(f"<div class='prediction-bad'>{label}</div>", unsafe_allow_html=True)
                else:
                    label = "Fresh"
                    confidence = fresh_prob * 100
                    st.markdown(f"<div class='prediction-good'>{label}</div>", unsafe_allow_html=True)

                st.progress(int(confidence))
                st.write(f"Confidence Score: {confidence:.2f}%")

                df = pd.DataFrame({
                    "Class": ["Fresh", "Rotten"],
                    "Probability": [fresh_prob*100, rotten_prob*100]
                })

                st.plotly_chart(px.bar(df, x="Class", y="Probability", text="Probability"))

                pdf = generate_pdf(label, confidence)
                st.download_button("Download Report", pdf, file_name="report.pdf")

                if "prediction_history" not in st.session_state:
                    st.session_state.prediction_history = []

                st.session_state.prediction_history.append({
                    "Time": datetime.now(),
                    "Prediction": label,
                    "Confidence": confidence
                })

# ==========================================
# ANALYTICS PAGE
# ==========================================
elif page == " Analytics":

    st.title("Model Performance Analytics")

    if "prediction_history" in st.session_state:

        df = pd.DataFrame(st.session_state.prediction_history)

        st.dataframe(df)

        st.plotly_chart(px.pie(df, names="Prediction", title="Class Distribution"))
        st.plotly_chart(px.line(df, x="Time", y="Confidence", color="Prediction"))

    else:
        st.info("No prediction history available yet.")

# ==========================================
# ABOUT PAGE
# ==========================================
elif page == " About":

    st.title("System Overview")

    with st.expander("Project Description"):
        st.write("""
        This system is a Deep Learning-based Fruit Freshness Classification tool.
        It uses MobileNetV2 CNN architecture to analyze fruit images and classify them as Fresh or Rotten.

        The system integrates:
        - Computer Vision preprocessing
        - Deep Learning classification
        - Explainable AI (Grad-CAM)
        - Interactive Streamlit dashboard
        - PDF report generation

        It is designed for real-time smart agriculture and food quality inspection systems.
        """)

    with st.expander("Technologies"):
        st.write("""
        - TensorFlow / Keras
        - MobileNetV2
        - OpenCV
        - Streamlit
        - Plotly
        - ReportLab
        - Grad-CAM
        """)

    with st.expander("Future Work"):
        st.write("""
        - Multi-fruit classification
        - Disease detection
        - Edge deployment (mobile/IoT)
        - Real-time video processing
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
