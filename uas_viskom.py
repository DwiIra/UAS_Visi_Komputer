import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model (pretrained on COCO dataset)
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # small model, fast
    return model

model = load_model()
st.title("Deteksi Objek dengan Streamlit")
st.write("Aplikasi ini menjalankan deteksi objek secara real-time di browser menggunakan webcam.")

# Upload image or open webcam
option = st.radio("Pilih Sumber Gambar:", ("Webcam", "Upload Gambar"))

if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload file gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Diupload", use_column_width=True)
        results = model(image)
        results.render()
        st.image(results.ims[0], caption="Hasil Deteksi", use_column_width=True)
else:
    run = st.checkbox('Nyalakan Kamera')
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Gagal membaca dari kamera.")
            break

        # Deteksi objek
        results = model(frame)
        results.render()  # render() will draw bounding boxes to image
        frame = results.ims[0]

        # Konversi BGR ke RGB untuk Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()

