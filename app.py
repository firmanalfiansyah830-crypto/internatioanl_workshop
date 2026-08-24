import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Deteksi & Perhitungan Objek YOLO",
    layout="wide"
)

st.title("📦 Deteksi & Perhitungan Objek dengan YOLO")
st.write("Unggah gambar untuk mendeteksi objek dan menghitung jumlah tiap kodenya secara otomatis.")

# 2. Sidebar - Pengaturan Model
st.sidebar.header("Pengaturan Model")

# Pilihan bobot YOLO (bisa diganti ke yolov8m.pt atau model kustom)
model_type = st.sidebar.selectbox(
    "Pilih Model YOLO",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(model_name)

try:
    model = load_yolo_model(model_type)
    st.sidebar.success(f"Model {model_type} berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"Gagal memuat model: {e}")

# 3. Main Interface - Input Gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca Gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Deteksi Objek menggunakan YOLO
    with st.spinner("Mendeteksi objek..."):
        results = model.predict(source=img_array, conf=conf_threshold)

    # Mengambil hasil pendeteksian
    result = results[0]
    
    # 1. Gambar Hasil Bounding Box
    res_plotted = result.plot()
    res_image = Image.fromarray(res_plotted[..., ::-1]) # Konversi BGR ke RGB

    # 2. Menghitung Jumlah Objek Berdasarkan Kelas
    detected_classes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    counts = Counter(detected_classes)

    # 4. Tampilan Visual Hasil
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Visualisasi Deteksi")
        st.image(res_image, use_container_width=True)

    with col2:
        st.subheader("Statistik Hasil")
        total_objects = len(detected_classes)
        
        st.metric(label="Total Objek Terdeteksi", value=total_objects)
        st.divider()

        if counts:
            st.write("**Rincian Per Tipe Objek:**")
            
            # Menampilkan tabel statistik
            data_table = [{"Tipe Objek": k, "Jumlah": v} for k, v in counts.items()]
            st.dataframe(data_table, use_container_width=True)
            
            # Menampilkan diagram batang ringkas
            st.write("**Grafik Distribusi:**")
            st.bar_chart(counts)
        else:
            st.warning("Tidak ada objek yang terdeteksi dengan threshold confidence saat ini.")