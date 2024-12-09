import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from tqdm import tqdm
from modules.processing.filters import apply_filter  # Import filter function

def extract_respiration_signal(frames):
    """
    Extract respiration signal from webcam frames using MediaPipe.
    Args:
        frames (list): List of frames from webcam.
    Returns:
        np.ndarray: Processed respiration signal.
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        signals = []

        # Use tqdm to show progress for frame processing
        for frame in tqdm(frames, desc="Processing frames", unit="frame"):
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame and extract landmarks
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Access nose tip landmark using index
                    nose_tip = face_landmarks.landmark[1]  # Example: Index 1 for nose tip
                    signals.append(nose_tip.z)

        # Convert to numpy array for easier manipulation
        respiration_signal = np.array(signals)

        # Apply filter to clean the respiration signal (optional)
        filtered_respiration_signal = apply_filter(respiration_signal, lowcut=0.05, highcut=0.5, fs=30)

        return filtered_respiration_signal
