import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ..processing import respiration, tracking
from ..visualization import plotter

def process_webcam(model_path, max_seconds, x_size, y_size, shift_x, shift_y):
    """Fungsi utama untuk memproses webcam"""
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)
    
    # Setup pose landmarker
    pose_landmarker = tracking.setup_pose_landmarker(model_path)
    
    # Initialize data arrays
    timestamps = []
    y_positions = []
    
    try:
        # Process first frame
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame!")
            
        # Initialize tracking
        tracking_data = tracking.initialize_tracking(first_frame, pose_landmarker, 
                                                  x_size, y_size, shift_x, shift_y)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
                
            # Process frame for respiration
            frame, y_pos = respiration.process_frame(frame, tracking_data)
            
            if y_pos is not None:
                current_time = time.time() - start_time
                timestamps.append(current_time)
                y_positions.append(y_pos)
                
                # Overlay plot pada frame
                frame = plotter.overlay_plot_on_frame(frame, timestamps, y_positions)
            
            # Display frame
            cv2.imshow("Shoulder Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Plot hasil akhir
    plotter.plot_shoulder_movement(timestamps, y_positions)
        
    return timestamps, y_positions