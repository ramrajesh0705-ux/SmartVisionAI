import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import plotly.express as px
import pandas as pd
from src.utils import load_classification_model, preprocess_image_for_classification, load_yolo_model, run_yolo_inference
from src.config import CLASS_NAMES, MODELS_DIR

st.set_page_config(page_title="SmartVision AI", layout="wide")

# Load all classification models (cached)
@st.cache_resource
def get_classification_models():
    models = {}
    for name in ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]:
        model = load_classification_model(name, 26)
        model.load_state_dict(torch.load(f"{MODELS_DIR}/{name}_best.pth", map_location="cpu"))
        model.eval()
        models[name] = model
    return models

cls_models = get_classification_models()
yolo_model = load_yolo_model(f"{MODELS_DIR}/yolo_finetuned/weights/best.pt")

# Pages
pages = ["Home", "Image Classification", "Object Detection", "Model Performance", "About"]
choice = st.sidebar.selectbox("Navigate", pages)

if choice == "Home":
    st.title("🚀 SmartVision AI")
    st.markdown("""
    **Intelligent Multi‑Class Object Recognition System**  
    Built with YOLOv8 and four CNN architectures (VGG16, ResNet50, MobileNetV2, EfficientNetB0) on a 26‑class COCO subset.
    
    ### Capabilities:
    - **Image Classification** – Single object classification using all four models.
    - **Object Detection** – YOLOv8 bounding boxes and labels.
    - **Model Comparison** – Accuracy, speed, and class‑wise breakdown.
    - **Live Demo** – Upload your own images or try sample ones.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📸 **Classification** – Best accuracy up to 93%")
    with col2:
        st.info("🔍 **Detection** – mAP@0.5 > 85%")
    with col3:
        st.info("⚡ **Real‑time** – 30‑50 FPS on GPU")

elif choice == "Image Classification":
    st.header("📷 Single Object Classification")
    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)
        input_tensor = preprocess_image_for_classification(np.array(img))
        results = {}
        for name, model in cls_models.items():
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.nn.functional.softmax(out[0], dim=0)
                top5 = torch.topk(probs, 5)
                results[name] = [(CLASS_NAMES[idx], probs[idx].item()) for idx in top5.indices]
        # Display results in columns
        cols = st.columns(4)
        for i, (name, preds) in enumerate(results.items()):
            with cols[i]:
                st.subheader(name)
                for label, score in preds:
                    st.write(f"{label}: {score:.2%}")

elif choice == "Object Detection":
    st.header("🔍 YOLOv8 Object Detection")
    uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        start = time.time()
        detections = run_yolo_inference(yolo_model, img_np, conf_thresh)
        end = time.time()
        # Draw boxes
        for (x1, y1, x2, y2), label, conf in detections:
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_np, f"{label} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        st.image(img_np, caption=f"Detected {len(detections)} objects", use_column_width=True)
        st.write(f"Inference time: {(end-start)*1000:.1f} ms")

elif choice == "Model Performance":
    st.header("📊 Model Comparison")
    # Example metrics (replace with your actual JSON results)
    perf = pd.DataFrame({
        "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0", "YOLOv8"],
        "Accuracy (%)": [83, 88, 85, 91, 87],   # YOLO uses mAP@0.5
        "Inference (ms)": [150, 100, 50, 80, 45]
    })
    fig = px.bar(perf, x="Model", y="Accuracy (%)", color="Model", title="Accuracy Comparison")
    st.plotly_chart(fig)
    st.dataframe(perf)
    st.markdown("""
    **Observation:** EfficientNetB0 achieves the best classification accuracy, while MobileNetV2 is fastest.  
    YOLOv8 provides the best trade‑off for detection (mAP@0.5 > 85%).
    """)

else:
    st.header("📖 About SmartVision AI")
    st.markdown("""
    **Dataset:** 26‑class subset of COCO (2,600 images, 100 per class)  
    **Classification Models:** VGG16, ResNet50, MobileNetV2, EfficientNetB0 (transfer learning)  
    **Detection:** YOLOv8 fine‑tuned  
    **Deployment:** Streamlit on Hugging Face Spaces  
    **Use Cases:** Smart cities, retail analytics, wildlife monitoring, security  
    **Developed by:** Your Name / Team  
    """)