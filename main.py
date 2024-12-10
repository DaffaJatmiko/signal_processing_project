import os
from modules.utils import download_model, check_gpu
from modules.input.webcam import process_webcam
from modules.processing.respiration import get_initial_roi
from modules.visualization.plotter import plot_shoulder_movement


def main():
    try:
        # Download model
        model_path = download_model()

        # Check GPU availability
        gpu_type = check_gpu()

        # Process video from webcam
        timestamps, y_positions = process_webcam(
            model_path="models/pose_landmarker.task",
            max_seconds=30,  # Waktu maksimum untuk merekam (dalam detik)
            x_size=100,  # Ukuran ROI pada sumbu X
            y_size=100,  # Ukuran ROI pada sumbu Y
            shift_x=0,  # Perpindahan ROI pada sumbu X
            shift_y=0  # Perpindahan ROI pada sumbu Y
        )

        # Plot shoulder movement
        plot_shoulder_movement(timestamps, y_positions)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
