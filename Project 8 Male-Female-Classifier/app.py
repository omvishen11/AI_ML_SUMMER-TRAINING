import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="♂️♀️",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("Female_Male_model.pkl")

model = load_model()
IMG_SIZE = 64

st.title("♂️ Male vs ♀️Female Image Classifier")
st.write("Upload an image to predict whether it is a Male or Female.")

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image using Pillow
    image = Image.open(uploaded_file)

    # Convert to RGB (important if image is grayscale/RGBA)
    image = image.convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", width=300)

    # Resize image
    resized = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to NumPy array
    resized = np.array(resized)

    # -------------------------
    # CRITICAL FIX: Pixel Normalization & Formatting
    # -------------------------
    # 1. Scale pixel values to 0-1 range to match training
    resized = resized / 255.0 
    
    # 2. Flatten image array into a 1D vector (Size: 64 * 64 * 3 = 12288)
    flattened_img = resized.flatten()

    # 3. Reshape to a 2D array structure expected by scikit-learn: [1, 12288]
    input_data = flattened_img.reshape(1, -1)

    # -------------------------
    # Prediction
    # -------------------------
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Display prediction
    if prediction == 0:
        st.success(" Prediction: ♀️Female")
    else:
        st.success(" Prediction: ♂️Male ")

    # Display probabilities
    st.subheader("Prediction Confidence")

    st.write(f"♀️ Female Probability: **{probability[0] * 100:.2f}%**")
    st.write(f"♂️ Male Probability: **{probability[1] * 100:.2f}%**")