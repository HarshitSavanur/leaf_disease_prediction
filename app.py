import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Custom CSS
st.markdown("""
<style>
.title {
    font-size: 2.5em;
    font-weight: bold;
    text-align: center;
    color: #34a853;
    margin-bottom: 0.1em;
}
.subtitle {
    text-align: center;
    color: #aaa;
    font-size: 1.1em;
    margin-bottom: 2em;
}
.result-box {
    background-color: rgba(52, 168, 83, 0.15); /* soft green */
    border-left: 6px solid #34a853;
    padding: 1.2em;
    border-radius: 10px;
    font-size: 1.1em;
    margin-top: 1.5em;
    color: #e6e6e6;
}
img.rounded-img {
    border-radius: 15px;
    border: 2px solid #ccc;
}
.label {
    text-align: center;
    margin-top: 0.5em;
    font-weight: 500;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# Load model and class names
model = load_model("my_model.h5")
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Title
st.markdown('<div class="title">🌿 Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a leaf image or try a sample image below</div>', unsafe_allow_html=True)

# ---- Upload Section ----
st.subheader("📤 Upload Your Image")
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
# ---- Sample Images ----
st.subheader("🖼️ Try Sample Images")
sample_folder = "sample"
sample_files = [f for f in os.listdir(sample_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

def clean_label(filename):
    return filename.replace("_", " ").replace(".JPG", "").replace(".jpg", "").replace(".png", "").replace(".jpeg", "")

selected_sample = None
if sample_files:
    num_cols = 4
    for row_start in range(0, len(sample_files), num_cols):
        cols = st.columns(num_cols)
        for i in range(num_cols):
            idx = row_start + i
            if idx < len(sample_files):
                file = sample_files[idx]
                filepath = os.path.join(sample_folder, file)
                with cols[i]:
                    st.image(filepath, width=130, output_format="JPEG", use_container_width=True)
                    st.markdown(f'<div class="label">{clean_label(file)}</div>', unsafe_allow_html=True)
                    if st.button(f"Select", key=f"btn_{file}"):
                        selected_sample = filepath
else:
    st.warning("No sample images found in the 'sample' folder.")


# ---- Image Selection Logic ----
img_path = None
img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_path = uploaded_file.name
elif selected_sample:
    img = Image.open(selected_sample).convert("RGB")
    img_path = selected_sample

# ---- Prediction ----
if img:
    st.image(img, caption="Selected Leaf Image", use_column_width=True)

    img_resized = img.resize((256, 256))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        pred = model.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

    st.markdown(
        f'<div class="result-box">🩺 <strong>Prediction:</strong> {predicted_class}<br>📊 <strong>Confidence:</strong> {confidence:.2f}%</div>',
        unsafe_allow_html=True
    )
