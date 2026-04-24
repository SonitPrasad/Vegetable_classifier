from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "best.pt"

IMAGE_SIZE = 384

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()

# =========================
# APP UI
# =========================
st.set_page_config(
    page_title="Vegetable Classifier",
    page_icon="🥦",
    layout="centered"
)

st.title("Vegetable Classification")
st.write("Upload an image and the model will identify the vegetable.")

uploaded_file = st.file_uploader(
    "Upload vegetable image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Vegetable"):
        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        result = results[0]
        probs = result.probs

        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        class_name = result.names[class_id]

        st.subheader("Prediction Result")
        st.success(f"Vegetable: {class_name}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

        st.write("Class probabilities:")
        for idx, prob in enumerate(probs.data.tolist()):
            st.write(f"{result.names[idx]}: {prob * 100:.2f}%")
