import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import custom_object_scope
import cv2
from PIL import Image
import os
import tempfile
import model_utils

# Page Config
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #0e1117;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.info("Upload your trained model weights (.h5) and an input image to detect oil spills.")

# Model Uploader
model_file = st.sidebar.file_uploader("Upload Trained Model (.h5)", type=["h5"])

@st.cache_resource
def load_trained_model(model_path):
    with custom_object_scope({'precision': model_utils.precision, 
                              'recall': model_utils.recall, 
                              'f1_score': model_utils.f1_score}):
        # Try loading the full model first
        try:
            # compile=False is safer for inference to avoid metric mismatch issues
            model = load_model(model_path, compile=False)
            st.success(f"Loaded full model! Layers: {len(model.layers)}")
            return model
        except Exception as e:
            st.warning(f"Could not load full model directly: {e}")
            st.info("Attempting to reconstruct model architecture and load weights...")
            
            # Fallback: create architecture and load weights
            try:
                model = model_utils.xception(num_classes=8, activation_function='softmax')
                st.write(f"Reconstructed Architecture Layers: {len(model.layers)}")
                
                # Try loading weights
                try:
                    model.load_weights(model_path)
                    st.success("Loaded weights successfully!")
                    return model
                except Exception as e_load:
                    st.error(f"Standard load_weights failed: {e_load}")
                    st.info("Trying to load weights with by_name=True (partial load)...")
                    # Last resort: partial load
                    model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    st.warning("Loaded weights partially. Predictions might be inaccurate if critical layers are missing.")
                    return model
                    
            except Exception as e2:
                 st.error(f"Failed to rebuild model: {e2}")
                 return None

if model_file:
    # Save to temp file because load_model expects a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
        tmp_file.write(model_file.getvalue())
        tmp_model_path = tmp_file.name
    
    with st.spinner("Loading Model..."):
        model = load_trained_model(tmp_model_path)
    
    if model:
        st.sidebar.success("Model loaded successfully!")
    
    # Cleanup
    os.remove(tmp_model_path)
else:
    model = None
    st.sidebar.warning("Please upload a .h5 model file to proceed with detection.")

# Main Interface
st.title("🌊 Oil Spill Detection System")
st.markdown("---")

col1, col2 = st.columns(2)

uploaded_image = st.file_uploader("Upload Input Image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_image:
    # Read Image
    image = Image.open(uploaded_image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image)
    
    with col1:
        st.subheader("Input Image")
        st.image(image_np, use_column_width=True)

    if model:
        if st.button("Detect Oil Spills"):
            with st.spinner("Processing..."):
                # Preprocess
                # Resize to 256x256 as required by the model
                img_resized = cv2.resize(image_np, (256, 256))
                
                # Preprocess input (normalization expected by Xception)
                # The notebook used: images_process = preprocess_input(images)
                # Ensure dimensions are (1, 256, 256, 3)
                img_batch = np.expand_dims(img_resized, axis=0)
                img_preprocessed = preprocess_input(img_batch)
                
                # Predict
                prediction = model.predict(img_preprocessed)
                
                # Post-process
                # Prediction shape: (1, 256, 256, 8) assuming categorical
                # or (1, 256, 256, 1) if binary
                
                output_shape = prediction.shape[-1]
                
                if output_shape > 1:
                    # Categorical: Mask is index of max class
                    mask = np.argmax(prediction, axis=-1)[0]
                    # Map to colors or just display raw
                    # Simple visualization: normalized grayscale
                    mask_vis = (mask / (output_shape - 1) * 255).astype(np.uint8)
                else:
                    # Binary
                    mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
                    mask_vis = mask
                
                # Resize mask back to original size for display overlay? 
                # Or just show 256x256 result. Showing 256x256 is safer/truer to model output.
                
                # Apply colormap for better visualization if categorical
                if output_shape > 1:
                    mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
                    # Convert BGR to RGB
                    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
                else:
                    mask_colored = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)

                with col2:
                    st.subheader("Detection Result")
                    st.image(mask_colored, caption="Predicted Mask", use_column_width=True)
                    
                    # Create an overlay
                    img_resized_rgb = cv2.resize(image_np, (256, 256))
                    overlay = cv2.addWeighted(img_resized_rgb, 0.6, mask_colored, 0.4, 0)
                    st.image(overlay, caption="Overlay on Input", use_column_width=True)
    else:
        st.info("Upload a model to see the output.")

else:
    st.info("Upload an image to get started.")
