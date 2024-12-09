import cv2
from tqdm import tqdm  # Import tqdm untuk progress bar


def capture_webcam(frame_count=300):
    """
    Capture frames from webcam stream.
    Args:
        frame_count (int): Number of frames to capture.
    Returns:
        list: List of frames captured from webcam.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    frames = []

    # Use tqdm to show progress for frame capture
    for _ in tqdm(range(frame_count), desc="Capturing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames