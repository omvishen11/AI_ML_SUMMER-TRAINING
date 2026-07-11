import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Configure the Streamlit page
st.set_page_config(page_title="Covid-19 X-Ray Detector", page_icon="🫁", layout="centered")

st.title("🫁 Covid-19 Chest X-Ray Detector")
st.write("Upload a chest X-ray image to detect potential signs of Covid-19.")

# Load the trained Keras model
# We use @st.cache_resource so it only loads once and stays in memory
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model1.keras')

with st.spinner("Loading model..."):
    model = load_model()

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image via Pillow and ensure it's in RGB format
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)
    
    st.write("Analyzing...")
    
    # --- Preprocessing ---
    # Resize to the target size used during training
    img_resized = image.resize((299, 299))
    
    # Convert image to numpy array
    img_array = np.array(img_resized)
    
    # Rescale pixel values exactly as done in the training ImageDataGenerator
    img_array = img_array / 255.0
    
    # Expand dimensions to match the batch format expected by the model: (1, 299, 299, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- Prediction ---
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    
    # Interpretation based on ["covid", "normal"] classes
    # Covid = 0, Normal = 1
    if confidence < 0.5:
        st.error(f"⚠️ **Result:** The model classified this X-ray as **COVID-19 Positive**.")
        st.write(f"**Confidence:** {(1 - confidence) * 100:.2f}%")
    else:
        st.success(f"✅ **Result:** The model classified this X-ray as **Normal**.")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

st.markdown("---")
st.markdown("*Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical diagnosis.*")