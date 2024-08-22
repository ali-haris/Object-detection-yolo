import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import ultralytics
import numpy as np
from PIL import Image
import tempfile
import time
import random
import os

# Set a fixed random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load the YOLO model
model = YOLO(r"your_model_path")  # Update with your model path

# Define color codes for objects
colors = {
    'orange': (255, 178, 29),
    'pineapple': (207, 210, 49)
}

def process_image(image):
    if image.shape[2] == 4:  # Check if the image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    results = model(image)
    return annotate_image(image, results), results

def annotate_image(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = box.conf[0].item()
            
            color = colors.get(class_name.lower(), (0, 255, 0))  # Default to green if not specified
            
            # Draw the bounding box with uniform thickness
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Adjusted thickness
            
            # Draw the label and confidence with proper font size inside the bounding box
            label = f"{class_name} {confidence:.2f}"
            print("********************************")
            print(label)
            print("********************************")
            font_scale = 1
            font_thickness = 2
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image, (x1, y1), (x1 + text_size[0], y1 - text_size[1] - 10), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)  # Adjusted font scale and thickness
    return image

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    progress_bar = st.progress(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, _ = process_image(frame)
        out.write(annotated_frame)
        
        current_frame += 1
        progress = current_frame / total_frames
        progress_bar.progress(progress)
    
    cap.release()
    out.release()

# Streamlit app
st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    annotated_image, results = process_image(image)
    st.image(annotated_image, caption='Annotated Image', use_column_width=True)
    
    # Display the number of detected objects and their names
    num_objects = sum(len(result.boxes) for result in results)
    detected_classes = [model.names[int(box.cls[0])] for result in results for box in result.boxes]
    st.write(f"Number of objects detected: {num_objects}")
    st.write(f"Detected objects: {', '.join(detected_classes)}")

uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    output_path = "output_" + os.path.basename(video_path)
    process_video(video_path, output_path)
    
    st.write("Processed video is ready for download.")
    with open(output_path, "rb") as file:
        st.download_button(label="Download Processed Video", data=file, file_name=output_path, mime="video/mp4")

# For debugging
st.write(f"Torch version: {torch.__version__}")
st.write(f"Ultralytics version: {ultralytics.__version__}")
