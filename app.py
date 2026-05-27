import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE = 224

# Change this to match the backbone used in train_model.py
# Options: 'mobilenetv2' | 'efficientnetb0' | 'resnet50' | 'vgg16' | 'none'
BACKBONE = 'mobilenetv2'

PREPROCESS_FN = {
    'mobilenetv2':    tf.keras.applications.mobilenet_v2.preprocess_input,
    'efficientnetb0': tf.keras.applications.efficientnet.preprocess_input,
    'resnet50':       tf.keras.applications.resnet50.preprocess_input,
    'vgg16':          tf.keras.applications.vgg16.preprocess_input,
    'none':           lambda x: x / 255.0,
}


# ─────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────
def process_image(image_bytes: bytes) -> tf.Tensor:
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    preprocess = PREPROCESS_FN.get(BACKBONE, PREPROCESS_FN['none'])
    image = preprocess(image)
    return tf.expand_dims(image, axis=0)


# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("dog_breed_model.h5")
    except Exception as e:
        st.error(f"Model file not found: {e}\nPlease run train_model.py first.")
        return None

@st.cache_resource
def load_breed_labels():
    try:
        return np.load("breed_labels.npy", allow_pickle=True)
    except Exception as e:
        st.error(f"Breed labels file not found: {e}\nPlease run train_model.py first.")
        return None


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dog Breed Detector",
    page_icon="🐾",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — paw-print tiled background
# ─────────────────────────────────────────────
# SVG paw print encoded as a CSS background data URI
PAW_SVG = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' "
    "viewBox='0 0 120 120'%3E"
    # main pad
    "%3Cellipse cx='60' cy='80' rx='18' ry='14' fill='rgba(255%2C255%2C255%2C0.06)'/%3E"
    # top-left toe
    "%3Cellipse cx='34' cy='56' rx='9' ry='11' fill='rgba(255%2C255%2C255%2C0.06)' transform='rotate(-20 34 56)'/%3E"
    # top-centre-left toe
    "%3Cellipse cx='50' cy='46' rx='9' ry='11' fill='rgba(255%2C255%2C255%2C0.06)' transform='rotate(-8 50 46)'/%3E"
    # top-centre-right toe
    "%3Cellipse cx='70' cy='46' rx='9' ry='11' fill='rgba(255%2C255%2C255%2C0.06)' transform='rotate(8 70 46)'/%3E"
    # top-right toe
    "%3Cellipse cx='86' cy='56' rx='9' ry='11' fill='rgba(255%2C255%2C255%2C0.06)' transform='rotate(20 86 56)'/%3E"
    "%3C/svg%3E"
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;900&family=Space+Mono:wght@700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Nunito', sans-serif;
}}

/* ── Paw-print background ── */
.stApp {{
    background-color: #1b3a2d;
    background-image:
        radial-gradient(ellipse at 20% 0%,  rgba(91,174,101,0.30) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 100%, rgba(44,120,60,0.25)  0%, transparent 55%),
        url("{PAW_SVG}");
    background-size: auto, auto, 120px 120px;
    min-height: 100vh;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.8rem; padding-bottom: 3rem; max-width: 780px; }}

/* ── Hero ── */
.hero {{
    text-align: center;
    padding: 2.4rem 1rem 1.4rem;
    margin-bottom: 1.8rem;
}}
.hero-icon {{
    font-size: 4rem;
    line-height: 1;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 20px rgba(160,230,120,0.6));
    display: inline-block;
    animation: bounce 2.4s ease-in-out infinite;
}}
@keyframes bounce {{
    0%,100% {{ transform: translateY(0) rotate(-5deg);  }}
    50%      {{ transform: translateY(-10px) rotate(5deg); }}
}}
.hero h1 {{
    font-size: 2.9rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    margin: 0 0 0.4rem;
    color: #fff;
    text-shadow: 0 2px 24px rgba(0,0,0,0.4);
}}
.hero h1 span {{
    background: linear-gradient(90deg, #a8e063, #56ab2f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero p {{
    color: rgba(255,255,255,0.55);
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}}

/* ── Upload card ── */
.upload-card {{
    background: rgba(255,255,255,0.06);
    border: 2px dashed rgba(168,224,99,0.4);
    border-radius: 22px;
    padding: 1.8rem 1.5rem;
    margin-bottom: 1.4rem;
    transition: border-color 0.3s, background 0.3s;
}}
.upload-card:hover {{
    border-color: rgba(168,224,99,0.75);
    background: rgba(255,255,255,0.09);
}}
.upload-label {{
    color: rgba(255,255,255,0.6);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}}

/* ── Image preview ── */
.img-frame {{
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid rgba(168,224,99,0.3);
    box-shadow: 0 12px 48px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04);
    margin-bottom: 1.6rem;
}}

/* ── Bone divider ── */
.bone-divider {{
    text-align: center;
    font-size: 1.4rem;
    margin: 0.6rem 0 1.4rem;
    opacity: 0.45;
    letter-spacing: 6px;
}}

/* ── Result card ── */
.result-wrap {{
    position: relative;
    border-radius: 22px;
    padding: 2px;
    background: linear-gradient(135deg, #a8e063, #56ab2f, #1a6b2f);
    box-shadow: 0 8px 40px rgba(86,171,47,0.25);
    animation: glow-pulse 3s ease-in-out infinite;
}}
@keyframes glow-pulse {{
    0%,100% {{ box-shadow: 0 8px 40px rgba(86,171,47,0.25); }}
    50%      {{ box-shadow: 0 8px 56px rgba(168,224,99,0.45); }}
}}
.result-inner {{
    background: rgba(14,40,22,0.92);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 2.2rem 2rem;
    text-align: center;
}}
.result-paw {{
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    display: inline-block;
    animation: spin-once 0.6s ease-out;
}}
@keyframes spin-once {{
    from {{ transform: rotate(-20deg) scale(0.7); opacity: 0; }}
    to   {{ transform: rotate(0deg)  scale(1);   opacity: 1; }}
}}
.result-label {{
    color: rgba(168,224,99,0.65);
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}
.result-breed {{
    font-size: 2.2rem;
    font-weight: 900;
    color: #fff;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 20px rgba(168,224,99,0.35);
    line-height: 1.15;
}}

/* ── File uploader polish ── */
[data-testid="stFileUploader"] {{
    background: transparent;
}}
[data-testid="stFileUploader"] section {{
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    border: 1px solid rgba(168,224,99,0.2);
}}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: #a8e063 !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
model         = load_model()
unique_breeds = load_breed_labels()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🐶</div>
    <h1>Dog Breed <span>Detector</span></h1>
    <p>Upload a photo and let AI identify the breed instantly</p>
</div>
""", unsafe_allow_html=True)

if model is None or unique_breeds is None:
    st.error("⚠️ Model or labels not found. Run `python train_model.py` first.")
    st.stop()

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<p class="upload-label">📁 &nbsp; Upload your dog photo</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICT & DISPLAY
# ─────────────────────────────────────────────
if uploaded_file is not None:

    # Bug 1 fix: reset pointer after reading bytes
    image_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
    st.image(image, width=740)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="bone-divider">🦴 &nbsp; 🐾 &nbsp; 🦴</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Sniffing out the breed…"):
        image_tensor    = process_image(image_bytes)
        prediction      = model.predict(image_tensor, verbose=0)
        predicted_idx   = int(np.argmax(prediction[0]))
        predicted_breed = str(unique_breeds[predicted_idx]).replace("_", " ").title()

    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-inner">
            <div class="result-paw">🐾</div>
            <div class="result-label">Detected Breed</div>
            <div class="result-breed">{predicted_breed}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)