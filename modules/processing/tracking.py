# modules/processing/tracking.py

import cv2
import numpy as np
import mediapipe as mp

def setup_pose_landmarker(model_path):
    """Setup MediaPipe pose landmarker"""
    # Ini adalah bagian dari kode webcam.py yang sebelumnya untuk setup landmarker
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return PoseLandmarker.create_from_options(options)

def get_initial_roi(frame, pose_landmarker, x_size, y_size, shift_x, shift_y):
    """
    Ini adalah fungsi yang sebelumnya ada di webcam.py untuk mendapatkan ROI
    berdasarkan posisi bahu (landmarks 11 dan 12)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    
    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )
    
    # Detect landmarks
    detection_result = pose_landmarker.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in frame!")
        
    landmarks = detection_result.pose_landmarks[0]
    
    # Get shoulder positions (landmarks 11 dan 12)
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    # Calculate center point between shoulders
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
    
    # Apply shifts
    center_x += shift_x
    center_y += shift_y
    
    # Calculate ROI boundaries
    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)
    
    return (left_x, top_y, right_x, bottom_y)

def initialize_features(roi, left_x, top_y):
    """
    Ini adalah bagian dari kode yang sebelumnya ada di dalam loop webcam.py
    untuk mendeteksi titik-titik yang akan di-track
    """
    features = cv2.goodFeaturesToTrack(roi, 
                                     maxCorners=60,
                                     qualityLevel=0.15,
                                     minDistance=3,
                                     blockSize=7)
    
    if features is not None:
        features = np.float32(features)
        # Adjust coordinates to full frame
        features[:,:,0] += left_x
        features[:,:,1] += top_y
        
    return features

def get_lk_params():
    """
    Parameter untuk Lucas-Kanade optical flow yang sebelumnya 
    didefinisikan di webcam.py
    """
    return dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

def initialize_tracking(frame, pose_landmarker, x_size, y_size, shift_x, shift_y):
    """
    Fungsi ini menggabungkan beberapa langkah inisialisasi yang sebelumnya 
    ada di webcam.py
    """
    # Get initial ROI
    roi_coords = get_initial_roi(frame, pose_landmarker, 
                               x_size, y_size, shift_x, shift_y)
    left_x, top_y, right_x, bottom_y = roi_coords
    
    # Initialize frame for optical flow
    old_frame = frame.copy()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Get initial ROI and detect features
    roi = old_gray[top_y:bottom_y, left_x:right_x]
    features = initialize_features(roi, left_x, top_y)
    
    # Return all tracking data in a dictionary
    return {
        'roi_coords': roi_coords,
        'old_gray': old_gray,
        'features': features,
        'lk_params': get_lk_params()
    }