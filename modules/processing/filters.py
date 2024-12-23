from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Create gentler Butterworth bandpass filter
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, fs=30):
    """
    Apply gentler bandpass filter for respiratory signal
    """
    # Menggunakan range yang lebih lebar untuk menangkap lebih banyak informasi
    lowcut = 0.1    # 6 breaths per minute
    highcut = 0.5   # 30 breaths per minute
    
    # Menurunkan order filter agar tidak terlalu agresif
    b, a = butter_bandpass(lowcut, highcut, fs, order=2)
    
    # Aplikasikan filter
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal

def preprocess_signal(y_positions, window=15):
    """
    Preprocess signal dengan pendekatan yang lebih lembut
    """
    y_array = np.array(y_positions)
    
    # Hapus outlier dengan metode yang lebih lembut
    percentile_5 = np.percentile(y_array, 5)
    percentile_95 = np.percentile(y_array, 95)
    mask = (y_array >= percentile_5) & (y_array <= percentile_95)
    y_cleaned = y_array[mask]
    
    # Normalisasi yang lebih sederhana
    y_normalized = y_cleaned - np.mean(y_cleaned)
    
    # Gunakan Savitzky-Golay filter untuk smoothing yang lebih baik
    y_smoothed = savgol_filter(y_normalized, 
                             window_length=15,  # window harus ganjil
                             polyorder=3)       # polynomial order
    
    return y_smoothed

def estimate_respiratory_rate(filtered_signal, timestamps):
    """
    Estimasi respiratory rate dengan parameter yang lebih sesuai
    """
    if len(filtered_signal) < 2:
        return 0, []
    
    # Deteksi puncak dengan parameter yang lebih longgar
    peaks, _ = find_peaks(filtered_signal,
                         distance=15,      # Minimal 15 sampel antar puncak
                         prominence=0.05,   # Prominence threshold yang lebih kecil
                         height=None)       # Tidak ada batasan tinggi
    
    if len(peaks) < 2:
        return 0, peaks
    
    # Hitung rate pernapasan
    peak_times = timestamps[peaks]
    intervals = np.diff(peak_times)
    
    # Filter interval yang masuk akal (2-8 detik per nafas)
    valid_intervals = intervals[(intervals >= 2) & (intervals <= 8)]
    
    if len(valid_intervals) > 0:
        mean_interval = np.mean(valid_intervals)
        breaths_per_minute = 60.0 / mean_interval
    else:
        breaths_per_minute = 0
    
    return breaths_per_minute, peaks