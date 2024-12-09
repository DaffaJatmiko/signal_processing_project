import cv2
import numpy as np
from modules.processing.filters import apply_filter


def extract_rppg(frames, roi_coords=None):
    """
    Extract rPPG signal from frames using color intensity variation in ROI.
    Args:
        frames (list): List of frames captured from webcam.
        roi_coords (tuple): Coordinates for Region of Interest (x, y, w, h).
    Returns:
        np.ndarray: Extracted rPPG signal.
    """
    signals = []
    for frame in frames:
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Define ROI (e.g., forehead area)
        if roi_coords is None:
            h, w, _ = rgb_frame.shape
            x, y, w, h = w // 3, h // 6, w // 3, h // 6  # Example: Center top region
        else:
            x, y, w, h = roi_coords

        roi = rgb_frame[y:y + h, x:x + w]

        # Calculate mean intensity of each channel
        mean_rgb = np.mean(roi, axis=(0, 1))  # Shape: [3] (R, G, B)
        signals.append(mean_rgb)

    # Convert to numpy array and separate channels
    signals = np.array(signals)
    r_signal = signals[:, 0]
    g_signal = signals[:, 1]
    b_signal = signals[:, 2]

    # Apply filtering (optional)
    filtered_signal = apply_filter(g_signal, lowcut=0.7, highcut=3.0, fs=30)  # Filtering green channel
    return filtered_signal
