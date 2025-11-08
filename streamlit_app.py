import streamlit as st
from src.vlm_model import VisionLanguageModel
from PIL import Image
import tempfile

st.set_page_config(page_title="Vision Agent", page_icon="🧠", layout="wide")

st.title("🧠 Vision Agent – Image Captioning AI")
st.write("Upload any image and let the AI describe it automatically!")

# Upload image
uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Initialize model only once
@st.cache_resource
def load_model():
    return VisionLanguageModel()

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Generate Description"):
        with st.spinner("Analyzing the image... please wait..."):
            vlm = load_model()

            # Save temp image for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name)
                caption = vlm.describe_image(tmp_file.name)

            st.success("✅ Caption Generated!")
            st.markdown(f"### 📝 Description: `{caption}`")
