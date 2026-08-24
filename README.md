# internatioanl_workshop
# 📦 YOLO Object Detection & Counting App

Aplikasi berbasis web menggunakan **Streamlit** dan **YOLOv8 (Ultralytics)** untuk mendeteksi objek pada gambar secara real-time, mengklasifikasikannya, dan menghitung jumlah tiap kodenya secara otomatis.

---

## 🚀 Fitur Utama

- **Pilihan Model YOLOv8**: Mendukung berbagai ukuran model (`yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`).
- **Pengaturan Confidence Threshold**: Geser slider untuk menyesuaikan tingkat sensitivitas deteksi.
- **Visualisasi Hasil**: Menampilkan gambar dengan *bounding box* dan label kelas secara jelas.
- **Statistik & Perhitungan Objek**:
  - Ringkasan total objek terdeteksi (*Metrics*).
  - Tabel rincian jumlah berdasarkan tipe/kategori objek.
  - Grafik batang (*Bar Chart*) distribusi jumlah objek.

---

## 📂 Struktur Proyek

```text
.
├── app.py              # Script utama aplikasi Streamlit
├── requirements.txt    # Daftar pustaka / dependensi Python
└── README.md           # Dokumentasi proyek
