import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set environment variables to reduce TensorFlow log verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Eye Gender Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_FILENAME = 'model.keras'

# ==========================================
# 2. Model Architecture & Loading
# ==========================================
def create_model_structure():
    """Generates the baseline CNN architecture."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(299, 299, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

@st.cache_resource(show_spinner="Loading AI Model...")
def load_trained_model():
    """Loads the model from disk and caches it in memory."""
    if os.path.exists(MODEL_FILENAME):
        try:
            return tf.keras.models.load_model(MODEL_FILENAME)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# ==========================================
# 3. Sidebar Configuration
# ==========================================
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.write("Configure model options and check status.")
    
    # Model Status
    if os.path.exists(MODEL_FILENAME):
        st.success("✅ Active Model: `model.keras`")
    else:
        st.warning("⚠️ No model found in workspace!")
        
        # Initialize Baseline
        if st.button("🔨 Generate Baseline Model", use_container_width=True):
            with st.spinner("Generating and compiling CNN..."):
                dummy_model = create_model_structure()
                dummy_model.save(MODEL_FILENAME)
                st.cache_resource.clear() # Reset cache so the new model loads
                st.rerun()

    st.divider()
    
    # Custom Model Upload
    st.subheader("Upload Custom Model")
    uploaded_model = st.file_uploader("Upload a .keras or .h5 file", type=["keras", "h5"])
    if uploaded_model:
        with open(MODEL_FILENAME, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.cache_resource.clear() # Force model reload
        st.success("✅ Custom model loaded!")
        st.rerun()

# Load the model into memory
model = load_trained_model()

# ==========================================
# 4. Main Interface
# ==========================================
st.title("👁️ Eye Gender Classifier")
st.write("Upload a photograph of a human eye to predict if it belongs to a Male or Female using a Deep Learning CNN.")

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📸 Upload Eye Photograph")
    uploaded_image = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Eye Photograph', use_container_width=True)

with col2:
    st.subheader("🔮 Prediction & Insights")
    
    if uploaded_image is None:
        st.info("👈 Upload an image on the left to see predictions.")
    elif model is None:
        st.error("⚠️ No model loaded. Generate or upload a model in the sidebar.")
    else:
        with st.spinner("Analyzing eye details..."):
            # Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            img_resized = image.resize((299, 299))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0) # Shape: [1, 299, 299, 3]
            
            # Predict
            result = model.predict(img_array)
            confidence = float(result[0][0])
            
            is_male = confidence > 0.5
            class_confidence = confidence if is_male else (1.0 - confidence)
            
            # Display Results natively
            if is_male:
                st.success("### ♂️ Prediction: Male Eye")
            else:
                st.error("### ♀️ Prediction: Female Eye")
                
            # Use native metric and progress bar for sleek UI
            st.metric(label="Confidence Level", value=f"{class_confidence * 100:.1f}%")
            
            st.write("**Raw Model Output (Sigmoid)**")
            st.progress(confidence)
            st.caption(f"Score: {confidence:.4f} (Scores > 0.5 indicate Male, < 0.5 indicate Female)")

st.divider()

# ==========================================
# 5. Architecture Details
# ==========================================
with st.expander("🔬 View Model Architecture & Preprocessing Details"):
    st.markdown("""
    ### Convolutional Neural Network (CNN) Structure
    The underlying Keras Sequential model consists of the following architecture:
    1. **Conv2D**: 32 filters, (3x3), ReLU -> **MaxPooling2D** (2,2)
    2. **Conv2D**: 32 filters, (3x3), ReLU -> **MaxPooling2D** (2,2)
    3. **Conv2D**: 64 filters, (3x3), ReLU -> **MaxPooling2D** (2,2)
    4. **Conv2D**: 128 filters, (3x3), ReLU -> **MaxPooling2D** (2,2)
    5. **Flatten**: Transforms 2D feature maps into a 1D vector.
    6. **Dense**: 128 units, ReLU activation.
    7. **Dense (Output)**: 1 unit, Sigmoid activation (Binary: `0 = Female`, `1 = Male`).
    
    ### Data Preprocessing
    * **Target Size**: 299x299 pixels (RGB)
    * **Normalization**: Pixel values rescaled to `[0, 1]` via `/ 255.0`
    """)