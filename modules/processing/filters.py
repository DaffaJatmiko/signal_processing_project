from scipy.signal import butter, filtfilt, find_peaks
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Membuat Butterworth bandpass filter
    
    Args:
        lowcut: Frekuensi cut-off bawah (Hz)
        highcut: Frekuensi cut-off atas (Hz)
        fs: Sampling frequency (Hz)
        order: Order filter
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, fs=30):
    """
    Menerapkan bandpass filter untuk sinyal respirasi
    
    Args:
        signal: Array sinyal y_positions
        fs: Sampling frequency (default 30 Hz dari webcam)
    
    Returns:
        filtered_signal: Sinyal yang sudah difilter
    """
    # Sesuaikan range frekuensi untuk pernapasan
    lowcut = 0.2  # Hz (12 breaths per minute)
    highcut = 0.5  # Hz (30 breaths per minute)
    
    # Buat dan terapkan filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=2)  # Kurangi order ke 2
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal

def estimate_respiratory_rate(filtered_signal, timestamps):
    """
    Mengestimasi respiratory rate dari sinyal yang sudah difilter
    
    Args:
        filtered_signal: Sinyal yang sudah difilter
        timestamps: Array waktu dalam detik
    
    Returns:
        breaths_per_minute: Estimasi napas per menit
        peaks: Indeks dari puncak-puncak sinyal
    """
    # Temukan puncak-puncak sinyal
    peaks, _ = find_peaks(filtered_signal, distance=15)  # Minimal 15 sampel antar puncak
    
    if len(peaks) < 2:
        return 0, peaks
    
    # Hitung rata-rana interval antar puncak
    peak_times = timestamps[peaks]
    intervals = np.diff(peak_times)
    mean_interval = np.mean(intervals)
    
    # Hitung napas per menit
    if mean_interval > 0:
        breaths_per_minute = 60.0 / mean_interval
    else:
        breaths_per_minute = 0
        
    return breaths_per_minute, peaks

def apply_moving_average(signal, window=5):
    """Apply moving average filter for smoothing"""
    return np.convolve(signal, np.ones(window)/window, mode='valid')

def preprocess_signal(y_positions, window=5):
    """Preprocess signal before bandpass filtering"""
    # Remove outliers
    y_array = np.array(y_positions)
    mean = np.mean(y_array)
    std = np.std(y_array)
    mask = np.abs(y_array - mean) <= 2 * std  # Keep points within 2 standard deviations
    y_cleaned = y_array[mask]
    
    # Apply moving average
    y_smoothed = apply_moving_average(y_cleaned, window)
    
    return y_smoothed

def estimate_respiratory_rate(filtered_signal, timestamps):
    """
    Mengestimasi respiratory rate dari sinyal yang sudah difilter
    
    Args:
        filtered_signal: Sinyal yang sudah difilter
        timestamps: Array waktu dalam detik
    
    Returns:
        breaths_per_minute: Estimasi napas per menit
        peaks: Indeks dari puncak-puncak sinyal
    """
    # Temukan puncak-puncak sinyal
    peaks, _ = find_peaks(filtered_signal, distance=15)  # Minimal 15 sampel antar puncak
    
    if len(peaks) < 2:
        return 0, peaks
    
    # Hitung rata-rata interval antar puncak
    peak_times = timestamps[peaks]
    intervals = np.diff(peak_times)
    mean_interval = np.mean(intervals)
    
    # Hitung napas per menit
    if mean_interval > 0:
        breaths_per_minute = 60.0 / mean_interval
    else:
        breaths_per_minute = 0
        
    return breaths_per_minute, peaks