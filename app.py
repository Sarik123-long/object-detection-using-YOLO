import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 pretrained model (COCO)
model = YOLO("yolov8n.pt")

# COCO vehicle + person classes
VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]
# 0-person, 1-bicycle, 2-car, 3-motorcycle, 5-bus, 7-truck

st.set_page_config(page_title="YOLO Vehicle Detection", layout="wide")

st.title("🚗 Vehicle Detection using YOLO (COCO Pretrained)")
st.write("Detect vehicles and persons in images or videos using YOLOv8")

# Sidebar
st.sidebar.header("Input Options")
option = st.sidebar.selectbox("Choose Input Type", ("Image", "Video"))

# ---------------- IMAGE DETECTION ----------------
if option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_array = np.array(image)

        results = model(img_array)
        annotated_img = results[0].plot()

        st.image(annotated_img, caption="Detected Objects", use_column_width=True)

# ---------------- VIDEO DETECTION ----------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{model.names[cls_id]} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, use_column_width=True)

        cap.release()

st.markdown("---")
st.markdown("Model: YOLOv8 (COCO Pretrained)")
st.markdown("Classes: person, bicycle, car, motorcycle, bus, truck")
