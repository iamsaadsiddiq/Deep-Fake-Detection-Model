import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
from fpdf import FPDF
from io import BytesIO
from streamlit_extras.let_it_rain import rain

# Load model once
@st.cache_resource

def load_deepfake_model():
    return load_model("xception_deepfake_stage1.h5")

model = load_deepfake_model()
IMAGE_SIZE = (224, 224)

# Session state init
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"
if "history" not in st.session_state:
    st.session_state["history"] = []

# Themes
THEMES = {
    "Light": {
        "primary": "#2563eb",
        "background": "#f8fafc",
        "secondary_bg": "#e0e7ef",
        "text": "#1e293b",
        "box_bg": "#ffffff",
        "accent": "#38bdf8"
    },
    "Dark": {
        "primary": "#38bdf8",
        "background": "#0f172a",
        "secondary_bg": "#1e293b",
        "text": "#f1f5f9",
        "box_bg": "#1e293b",
        "accent": "#2563eb"
    }
}

# Theme toggle UI
col1, col2 = st.columns([7, 2])
with col2:
    st.markdown("<div style='text-align:right;'>Theme:</div>", unsafe_allow_html=True)
    theme_choice = st.selectbox(
        "Choose Theme",
        ["🌞 Light", "🌙 Dark"],
        index=0 if st.session_state["theme"] == "Light" else 1,
        label_visibility="collapsed"
    )
    st.session_state["theme"] = "Light" if theme_choice.startswith("🌞") else "Dark"
THEME = THEMES[st.session_state["theme"]]

# CSS Styling
st.markdown(f"""
    <style>
    html, body, [class*="css"]  {{
        background-color: {THEME['background']} !important;
        color: {THEME['text']} !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    .stApp {{
        background-color: {THEME['background']} !important;
    }}
    .title {{
        text-align: center;
        font-size: 2.5em;
        color: {THEME['primary']};
        margin-bottom: 0.2em;
        font-weight: bold;
    }}
    .subtitle {{
        text-align: center;
        color: {THEME['accent']};
        font-size: 1.2em;
        margin-bottom: 1em;
    }}
    .box {{
        background-color: {THEME['box_bg']};
        padding: 1.5em;
        border-radius: 12px;
        border-left: 6px solid {THEME['primary']};
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin-top: 1.5em;
    }}
    .history-caption {{
        font-size: 0.9em;
        text-align: center;
        color: {THEME['text']};
    }}
    .credit-bar {{
        position: fixed;
        bottom: 0; left: 50%;
        transform: translateX(-50%);
        background-color: {THEME['primary']};
        color: #fff;
        padding: 8px 20px;
        font-size: 0.9em;
        border-radius: 10px 10px 0 0;
    }}
    </style>
""", unsafe_allow_html=True)

# Header
rain(emoji="🔍", font_size=20, falling_speed=5, animation_length="infinite")
st.markdown('<div class="title">Deepfake Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to detect whether it’s Real or Deepfake</div>', unsafe_allow_html=True)
st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# PDF report

def generate_pdf(label, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Deepfake Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="""
This AI model predicts whether the uploaded image is real or deepfake based on facial inconsistencies.
Model: Xception CNN
Indicators: Pixel-level artifacts, texture anomalies, asymmetry, etc.
    """)
    return BytesIO(pdf.output(dest='S').encode('latin-1'))

# Prediction
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        time.sleep(1.5)
        prediction = model.predict(img_batch)[0][0]
        confidence = prediction * 100 if prediction < 0.5 else (1 - prediction) * 100
        label = "Deepfake" if prediction < 0.5 else "Real"
        color = THEME["primary"] if label == "Deepfake" else THEME["accent"]

    st.session_state["history"].append({
        "label": label,
        "confidence": confidence,
        "image": image.copy(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    st.markdown(f"""
    <div class='box'>
        <h3 style='color:{color};'>Prediction: {label}</h3>
        <p style='font-size: 1.2em; color:{color};'>Confidence: {confidence:.2f}%</p>
        <p>The model examined facial patterns using CNN layers trained for forgery detection.</p>
    </div>
    """, unsafe_allow_html=True)

    pdf_buffer = generate_pdf(label, confidence)
    st.download_button("📄 Download PDF Report", data=pdf_buffer, file_name="deepfake_report.pdf", mime="application/pdf")

# History
if st.session_state["history"]:
    st.markdown("## Recent Predictions")
    cols = st.columns(5)
    for i, item in enumerate(reversed(st.session_state["history"][-5:])):
        with cols[i % 5]:
            st.image(item["image"].resize((150, 150)), caption=f"{item['label']} ({item['confidence']:.1f}%)\n{item['timestamp']}", width=150)

# Sidebar
st.sidebar.title("What are Deepfakes?")
st.sidebar.info("""
Deepfakes are synthetic media generated using AI. This tool helps detect such manipulated images.
Use it for education, awareness, and research on misinformation.
""")
st.sidebar.markdown("---")
st.sidebar.success("Built for Ma'am Aima Munir")

# Footer
st.markdown('<div class="credit-bar">Created by Muhammad Saad</div>', unsafe_allow_html=True)
