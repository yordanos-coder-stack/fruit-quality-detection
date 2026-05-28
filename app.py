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

    # MobileNetV2 base model
    base_model = model.layers[1]

    # Last conv layer
    last_conv_layer = base_model.get_layer(
        last_conv_layer_name
    )

    # Conv model
    last_conv_layer_model = tf.keras.Model(
        base_model.inputs,
        last_conv_layer.output
    )

    # Classifier model
    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    # Add top classifier layers
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

    # Gradient computation
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

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Features",
        "Workflow",
        "Samples"
    ])

    with tab1:

        st.write("""
          Real-time prediction  
          Explainable AI (Grad-CAM)  
          PDF report generation  
          CSV download  
          Interactive analytics  
          Webcam prediction  
        """)

    with tab2:

        workflow = pd.DataFrame({
            "Step": [
                "Upload Image",
                "Preprocessing",
                "CNN Prediction",
                "Grad-CAM",
                "Final Result"
            ],
            "Order": [1,2,3,4,5]
        })

        fig = px.line(
            workflow,
            x="Order",
            y="Order",
            text="Step",
            markers=True,
            title="System Workflow"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab3:

        st.image(
            [
                "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce",
                "https://images.unsplash.com/photo-1574226516831-e1dff420e37f"
            ],
            width=300
        )

# ==========================================
# PREDICTION PAGE
# ==========================================
elif page == " Prediction":

    st.title(" Fruit Freshness Prediction")

    prediction_method = st.radio(
        "Select Input Method",
        [
            "Upload Image",
            "Use Camera"
        ]
    )

    image = None

    # ==========================================
    # IMAGE INPUT
    # ==========================================
    if prediction_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload Fruit Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

    else:

        camera_image = st.camera_input(
            "Take a Picture"
        )

        if camera_image:

            image = Image.open(
                camera_image
            ).convert("RGB")

    # ==========================================
    # PREDICTION
    # ==========================================
    if image is not None:

        col1, col2 = st.columns([1,1])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            with st.spinner(
                "Analyzing Fruit Quality..."
            ):

                processed = preprocess_image(
                    image
                )

                prediction = model.predict(
                    processed
                )

                rotten_prob = float(
                    prediction[0][0]
                )

                fresh_prob = 1 - rotten_prob

                # ==========================================
                # CLASSIFICATION
                # ==========================================
                if rotten_prob > confidence_threshold:

                    label = "Rotten "

                    confidence = rotten_prob * 100

                    st.markdown(f"""
                    <div class="prediction-bad">
                    {label}
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    label = "Fresh "

                    confidence = fresh_prob * 100

                    st.markdown(f"""
                    <div class="prediction-good">
                    {label}
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(
                    int(confidence)
                )

                st.write(
                    f"### Confidence: {confidence:.2f}%"
                )

                # ==========================================
                # GAUGE CHART
                # ==========================================
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={
                            'text': "Confidence Score"
                        },
                        gauge={
                            'axis': {
                                'range': [0,100]
                            }
                        }
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

                # ==========================================
                # BAR CHART
                # ==========================================
                prob_df = pd.DataFrame({
                    "Class": [
                        "Fresh",
                        "Rotten"
                    ],
                    "Probability": [
                        fresh_prob * 100,
                        rotten_prob * 100
                    ]
                })

                bar_fig = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    text="Probability",
                    title="Prediction Probabilities"
                )

                st.plotly_chart(
                    bar_fig,
                    use_container_width=True
                )

                # ==========================================
                # EXPLAINABLE AI
                # ==========================================
                st.subheader(
                    " Explainable AI (Grad-CAM)"
                )

                try:

                    heatmap = make_gradcam_heatmap(
                        processed,
                        model
                    )

                    # Resize heatmap
                    heatmap = cv2.resize(
                        heatmap,
                        (
                            image.size[0],
                            image.size[1]
                        )
                    )

                    heatmap = np.uint8(
                        255 * heatmap
                    )

                    # Color map
                    heatmap = cv2.applyColorMap(
                        heatmap,
                        cv2.COLORMAP_JET
                    )

                    original = np.array(image)

                    # Overlay heatmap
                    superimposed_img = cv2.addWeighted(
                        original,
                        0.6,
                        heatmap,
                        0.4,
                        0
                    )

                    fig, ax = plt.subplots(
                        figsize=(6,6)
                    )

                    ax.imshow(
                        cv2.cvtColor(
                            superimposed_img.astype(
                                "uint8"
                            ),
                            cv2.COLOR_BGR2RGB
                        )
                    )

                    ax.axis("off")

                    st.pyplot(fig)

                except Exception as e:

                    st.error(
                        f"Grad-CAM Error: {e}"
                    )

                # ==========================================
                # DOWNLOAD CSV
                # ==========================================
                result_df = pd.DataFrame({
                    "Prediction": [label],
                    "Confidence": [
                        f"{confidence:.2f}%"
                    ]
                })

                csv = result_df.to_csv(
                    index=False
                ).encode('utf-8')

                st.download_button(
                    label=" Download CSV Result",
                    data=csv,
                    file_name="prediction_result.csv",
                    mime="text/csv"
                )

                # ==========================================
                # DOWNLOAD PDF
                # ==========================================
                pdf_file = generate_pdf(
                    label,
                    confidence
                )

                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_file,
                    file_name="fruit_prediction_report.pdf",
                    mime="application/pdf"
                )

                # ==========================================
                # SAVE HISTORY
                # ==========================================
                history = {
                    "Time": datetime.now(),
                    "Prediction": label,
                    "Confidence": round(
                        confidence,
                        2
                    )
                }

                if "prediction_history" not in st.session_state:

                    st.session_state.prediction_history = []

                st.session_state.prediction_history.append(
                    history
                )

# ==========================================
# ANALYTICS PAGE
# ==========================================
elif page == " Analytics":

    st.title(" Prediction Analytics")

    if "prediction_history" in st.session_state:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(history_df)

        # Pie Chart
        pie_fig = px.pie(
            history_df,
            names="Prediction",
            title="Prediction Distribution"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

        # Line Chart
        line_fig = px.line(
            history_df,
            x="Time",
            y="Confidence",
            color="Prediction",
            markers=True,
            title="Confidence Trend"
        )

        st.plotly_chart(
            line_fig,
            use_container_width=True
        )

    else:

        st.info(
            "No predictions available."
        )

# ==========================================
# ABOUT PAGE
# ==========================================
elif page == " About":

    st.title(" About This System")

    with st.expander(
        " Project Description"
    ):

        st.write("""
       This project is an AI-powered Fruit Freshness Classification System developed using Deep Learning and MobileNetV2. The system is designed to automatically determine whether a fruit is fresh or rotten by analyzing fruit images. It uses Artificial Intelligence (AI) and Computer Vision techniques to perform image-based classification accurately and efficiently.

       The core of the system is based on Deep Learning, a subset of Artificial Intelligence that enables computers to learn patterns and features automatically from data. Instead of manually defining image features such as color, texture, or shape, the deep learning model learns these characteristics directly from thousands of training images.

        The project uses MobileNetV2, a lightweight and efficient Convolutional Neural Network (CNN) architecture developed for image classification tasks. MobileNetV2 is particularly suitable for real-time applications because it provides high accuracy while requiring less computational power and memory compared to larger deep learning models.

        During training, the model learns important visual patterns associated with fresh and rotten fruits, such as:

            Color changes
            Texture differences
            Mold or dark spots
            Surface damage
            Shape irregularities

        When a user uploads a fruit image, the system first preprocesses the image by resizing and normalizing it. The processed image is then passed to the trained MobileNetV2 model, which predicts whether the fruit is fresh or rotten. The system also displays a confidence score indicating how certain the model is about its prediction.

        In addition, the project integrates Explainable AI using Grad-CAM (Gradient-weighted Class Activation Mapping). This technique generates heatmaps to highlight the image regions that most influenced the model’s prediction, improving transparency and interpretability.

        The entire system is deployed using Streamlit, an interactive Python framework for building machine learning web applications. The dashboard allows users to upload images, view predictions, analyze confidence scores, generate PDF reports, and visualize Explainable AI heatmaps in real time.""")

    with st.expander(
        " Technologies Used"
    ):

        st.write("""
        - TensorFlow / Keras
        - MobileNetV2 CNN
        - OpenCV
        - Streamlit
        - Plotly
        - ReportLab
        - Grad-CAM Explainable AI
        """)

    with st.expander(
        " Future Improvements"
    ):

        st.write("""
        - Multi-fruit classification
        - Disease detection
        - Real-time video analysis
        - Mobile application
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

