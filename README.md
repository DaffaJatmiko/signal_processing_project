# Signal Processing Project

Proyek ini menyediakan implementasi untuk ekstraksi sinyal Photoplethysmography jarak jauh (rPPG) dan sinyal respirasi menggunakan webcam. Program ini mendeteksi wajah manusia, mengekstrak wilayah minat (ROI) berdasarkan deteksi wajah, dan memproses data piksel untuk menghitung detak jantung serta sinyal respirasi.

## Instalasi

1. Pastikan Python 3.x terinstal di sistem Anda.
2. Clone Repository ini dengan:
   ```bash
   git clone https://github.com/DaffaJatmiko/signal_processing_project.git
   ```
3. Buat Virtual Environtment untuk project dengan:
   ```bash
   python -m venv desp-venv
   ```
   lalu aktivasi venv dengan:
   ```bash
   [path]/dsp-venv/Scripts/activate
   ```
4. Install dependensi dengan:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Penggunaan

### Menjalankan Program
1. Clone repositori dan pindah ke folder proyek.
2. Untuk menjalankan program, gunakan:
   ```bash
   python main.py ---> Untuk Menjalankan Proses Respiration Signal
   ```
   dan
   ```bash
   python mainrppg.py ---> Untuk Menjalankan Proses rPPG
   ```

### Fungsionalitas

#### rPPG
- Program menangkap video dari webcam.
- Memproses frame secara real-time untuk mendeteksi wajah dan mengekstrak sinyal intensitas piksel (saluran RGB).
- Setelah setiap 300 frame, program menghitung dan mencetak sinyal rPPG serta detak jantung.
- Setelah sesi satu menit, semua sinyal yang telah diproses digabungkan dan divisualisasikan dalam bentuk plot.

#### Respiration Signal
- Program menangkap video dari webcam.
- Memproses frame secara real-time untuk mendeteksi pergerakan bahu dan wajah dengan pemodelan pose_landmark.
- Program menghitung dan mencetak Respiration Signal serta detak jantung secara Real-Time.
- Setelah sesi diselesaikan, Respiration Signal yang telah diproses digabungkan dan divisualisasikan dalam bentuk plot.

### Kontrol
- Tekan `q` untuk keluar dari program kapan saja.

## Penjelasan Kode 
### rPPG
#### Modul dan Fungsi

##### `cpu_POS(signal, fps)`
Memproses sinyal RGB input menggunakan metode Plane-Orthogonal-to-Skin (POS) untuk mengekstraksi sinyal rPPG.

##### `process_rppg_signals(r_signal, g_signal, b_signal, fps)`
Melakukan penyaringan bandpass pada sinyal RGB dan menghitung detak jantung. Memvisualisasikan hasil jika diperlukan.

##### `main()`
Fungsi utama yang menginisialisasi umpan webcam, melakukan deteksi wajah, dan mengelola alur program secara keseluruhan.

#### Langkah-Langkah Pemrosesan Sinyal
1. Mengekstraksi ROI di sekitar wajah yang terdeteksi.
2. Menghitung rata-rata intensitas piksel untuk saluran R, G, dan B.
3. Menggunakan metode POS untuk mengekstraksi sinyal rPPG.
4. Menerapkan Chebyshev Filter untuk mengisolasi rentang frekuensi detak jantung.
5. Mendeteksi puncak dalam sinyal yang telah difilter untuk menghitung detak jantung.

### Respiration Signal
#### Modul dan Fungsi

##### `utils.py`
Mengidentifikasi jenis perangkat untuk kemudian akan mengunduh model `pose_landmark` yang sesuai dengan perangkat.

##### `webcam.py`
Mengaktifkan fungsi webcam serta mengaplikasikan model `pose_landmark` yang telah diunduh sebelumnya.

##### `tracking.py`
Menginisiasi proses tracking pada webcam saat pengaplikasian `pose_landmark`.

##### `filters.py`
Menginisiasi filter yang akan digunakan selama proses Respiration Signal.

##### `respiration.py`
Fungsi utama dalam membaca Respiration Signal.

##### `plotter.py`
Melakukan plotting pada akhir program.


#### Langkah-Langkah Pemrosesan Sinyal
1. Mengekstraksi ROI di sekitar bahu dan dagu yang terdeteksi.
2. Menghitung rata-rata intensitas pergeseran piksel antar titik dalam model `pose_landmark`.
3. Menerapkan filter bandpass untuk mengisolasi rentang frekuensi detak jantung.
4. Mendeteksi puncak dalam sinyal yang telah difilter untuk menghitung detak jantung.

## Hasil Output

### Sinyal rPPG dan Respirasi serta Detak Jantung
- Program mencetak detak jantung yang dihitung (BPM) setiap 300 frame untuk rPPG.
- Program mencetak Sinyal Pernapasan (Respiration Signal) pada akhir program.

### Visualisasi
- Pada akhir setiap sesi, sebuah plot akan ditampilkan, menunjukkan sinyal yang difilter dan puncak yang terdeteksi.

## Keterbatasan
- Berfungsi optimal dalam kondisi pencahayaan yang baik.

## Lisensi
Proyek ini bersifat open-source dan tersedia di bawah Lisensi MIT.

## Anggota Kelompok DSP (IF3024)
1. Ihsan Triyadi (121140163)
2. Andreas Sihotang (121140168)
3. Daffa Abdurrahman Jatmiko (121140181)