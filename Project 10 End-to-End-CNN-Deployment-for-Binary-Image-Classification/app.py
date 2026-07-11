import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Configure the page settings
st.set_page_config(
    page_title="CNN Image Classifier", 
    page_icon="🤖", 
    layout="centered"
)

st.title("Binary Image Classifier")
st.write("Upload an image, and the deployed Convolutional Neural Network (CNN) will classify it as Male or Female.")

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_classifier_model():
    return tf.keras.models.load_model('binary_image_classifier.keras')

try:
    model = load_classifier_model()
except Exception as e:
    st.error(f"Failed to load the model. Ensure 'binary_image_classifier.keras' is in the root directory. Error: {e}")
    st.stop()

# File upload widget
uploaded_file = st.file_uploader("Upload a facial image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    with st.spinner("Processing image through CNN..."):
        # Preprocessing pipeline matching the training environment
        img_rgb = image.convert('RGB')
        img_resized = img_rgb.resize((150, 150))
        
        # Convert to tensor and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  

        # Forward pass / Inference
        prediction = model.predict(img_array)
        pred_value = prediction[0][0]

        # Classification mapping (0 = Female, 1 = Male based on alphabetical directory flow)
        # if pred_value >= 0.5:
        #     label = "Male"
        #     confidence = pred_value * 100
        # else:
        #     label = "Female"
        #     confidence = (1.0 - pred_value) * 100
        # Swap the labels based on your model's actual output structure
        if pred_value >= 0.5:
            label = "Female"  
            confidence = pred_value * 100
        else:
            label = "Male"    
            confidence = (1.0 - pred_value) * 100

    # Output results
    st.divider()
    st.subheader("Inference Results")
    
    if label == "Male":
        st.info(f"**Predicted Class:** {label}  \n**Model Confidence:** {confidence:.2f}%")
    else:
        st.success(f"**Predicted Class:** {label}  \n**Model Confidence:** {confidence:.2f}%")