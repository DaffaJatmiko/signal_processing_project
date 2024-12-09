from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Create a Butterworth bandpass filter.
    Args:
        lowcut (float): Lower frequency cutoff.
        highcut (float): Upper frequency cutoff.
        fs (int): Sampling frequency.
        order (int): Order of the filter.
    Returns:
        tuple: Filter coefficients (b, a).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(signal, lowcut=0.1, highcut=1.0, fs=30, order=4):
    """
    Apply a Butterworth bandpass filter to the signal.
    Args:
        signal (np.ndarray): Input signal.
        lowcut (float): Lower frequency cutoff.
        highcut (float): Upper frequency cutoff.
        fs (int): Sampling frequency.
        order (int): Order of the filter.
    Returns:
        np.ndarray: Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
