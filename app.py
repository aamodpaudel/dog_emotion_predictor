import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import plotly.express as px
from models.resnet_model import DogEmotionResNet
from config import CLASSES, MODEL_SAVE_PATH, IMAGE_SIZE, DEVICE

# --- Page Configuration ---
st.set_page_config(
    page_title="Dog Emotion Analyzer | CareerlinkAI Tools",
    layout="wide"
)

# --- Professional Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; border: none; }
    .stButton>button:hover { background-color: #0056b3; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1c1e21; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- Core Logic ---
@st.cache_resource
def load_trained_model():
    """Loads the weights into the ResNet-18 architecture."""
    model = DogEmotionResNet(freeze_backbone=False)
    # map_location handles loading on CPU-only servers
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# --- UI Sidebar ---
with st.sidebar:
    st.title("Tool Settings")
    st.info("This interface utilizes a Fine-Tuned ResNet-18 architecture optimized for canine facial feature extraction.")
    st.divider()
    st.caption("Developed by CareerlinkAI Research and Development")

# --- Main Interface ---
st.title("Dog Emotion Recognition Engine")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Upload Analysis Subject")
    uploaded_file = st.file_uploader("Upload high-resolution image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Input Source', use_container_width=True)

with col2:
    st.subheader("Inference Analysis")
    
    if uploaded_file:
        if st.button('Execute Neural Analysis'):
            try:
                model = load_trained_model()
                img_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, prediction = torch.max(probabilities, 0)
                
                # --- Result Display ---
                emotion = CLASSES[prediction.item()].upper()
                conf_val = confidence.item()
                
                st.metric(label="Primary Classification", value=emotion, delta=f"{conf_val*100:.1f}% Confidence")
                
                # --- Probability Chart ---
                prob_df = pd.DataFrame({
                    'Emotion': [c.capitalize() for c in CLASSES],
                    'Probability': probabilities.cpu().numpy()
                }).sort_values('Probability', ascending=True)
                
                fig = px.bar(prob_df, x='Probability', y='Emotion', orientation='h', 
                             title="Classification Confidence Breakdown",
                             color='Probability', color_continuous_scale='Blues')
                fig.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Technical Summary ---
                st.markdown(f"""
                **Analysis Summary:**
                The model has identified visual markers consistent with a **{emotion.lower()}** state. 
                Statistical confidence for this classification is **{conf_val*100:.2f}%**.
                """)
                
            except Exception as e:
                st.error(f"System Error during inference: {str(e)}")
    else:
        st.info("Awaiting input image for processing.")

# --- Footer ---
st.markdown("---")
st.caption("© 2026 CareerlinkAI. All Rights Reserved. For academic and research purposes only.")