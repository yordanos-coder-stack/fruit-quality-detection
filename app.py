import streamlit as st
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
    page_title="AI Fruit Freshness Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CONSTANTS
# ==========================================
IMG_SIZE = 128

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    ["🏠 Home", " Prediction", " Analytics", " About"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.5, 0.01
)

st.sidebar.success("System Ready (Cloud Safe Mode)")

# ==========================================
# PREPROCESS IMAGE (SAFE)
# ==========================================
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ==========================================
# PDF REPORT
# ==========================================
def generate_pdf(label, confidence):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph(
        "Fruit Freshness Prediction Report",
        styles['Title']
    ))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        f"<b>Prediction:</b> {label}",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        f"<b>Confidence:</b> {confidence:.2f}%",
        styles['BodyText']
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==========================================
# HOME PAGE
# ==========================================
if page == "🏠 Home":

    st.markdown("""
    <div style="padding:30px;background:linear-gradient(135deg,#43cea2,#185a9d);
    color:white;text-align:center;border-radius:20px;">
        <h1>AI Fruit Freshness Detection System</h1>
        <p>Streamlit Cloud Compatible (No TensorFlow Version)</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "Demo Mode (No AI)")

    with col2:
        st.metric("Accuracy", "UI Simulation")

    with col3:
        st.metric("Classes", "Fresh / Rotten")

    st.info("This version simulates predictions for cloud deployment stability.")

# ==========================================
# PREDICTION PAGE (CLOUD SAFE AI SIMULATION)
# ==========================================
elif page == " Prediction":

    st.title("Fruit Freshness Prediction")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        processed = preprocess_image(image)

        # ======================================
        # SAFE SIMULATED PREDICTION (NO TF)
        # ======================================
        np.random.seed(abs(hash(str(image))) % 1000000)

        rotten_prob = float(np.random.uniform(0.1, 0.9))
        fresh_prob = 1 - rotten_prob

        if rotten_prob > confidence_threshold:
            label = "Rotten"
            confidence = rotten_prob * 100
            st.error(f"Prediction: {label}")
        else:
            label = "Fresh"
            confidence = fresh_prob * 100
            st.success(f"Prediction: {label}")

        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

        # ======================================
        # CHART
        # ======================================
        df = pd.DataFrame({
            "Class": ["Fresh", "Rotten"],
            "Probability": [fresh_prob * 100, rotten_prob * 100]
        })

        fig = px.bar(df, x="Class", y="Probability", text="Probability")
        st.plotly_chart(fig, use_container_width=True)

        # ======================================
        # PDF DOWNLOAD
        # ======================================
        pdf = generate_pdf(label, confidence)

        st.download_button(
            "Download PDF Report",
            pdf,
            file_name="fruit_report.pdf"
        )

        # ======================================
        # HISTORY
        # ======================================
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

    st.title("Prediction Analytics")

    if "history" in st.session_state:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df)

        fig = px.pie(df, names="Prediction", title="Distribution")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df, x="Time", y="Confidence", color="Prediction")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No data yet.")

# ==========================================
# ABOUT
# ==========================================
elif page == " About":

    st.title("About This System")

    st.write("""
This project is a Streamlit Cloud safe version of the Fruit Freshness Detection system.

### Changes made:
- Removed TensorFlow dependency
- Removed cv2 model dependency risk
- Added simulated AI prediction (safe mode)
- Fully deployable on Streamlit Cloud
- Preserves UI, analytics, and report generation
""")
