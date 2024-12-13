import time
import cv2
import numpy as np
import mediapipe as mp

def get_initial_roi(image, model_path, x_size, y_size, shift_x, shift_y):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Create a list to store the detection result
    detection_result = []
    
    def callback(result, output_image, timestamp_ms):
        detection_result.append(result)
    
    # Setup landmarker for initial detection
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose_detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = pose_detector.detect(mp_image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            raise ValueError("No pose detected!")

        landmarks = detection_result.pose_landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        center_x = int((left_shoulder.x + right_shoulder.x) * width / 2) + shift_x
        center_y = int((left_shoulder.y + right_shoulder.y) * height / 2) + shift_y

        left_x = max(0, center_x - x_size)
        right_x = min(width, center_x + x_size)
        top_y = max(0, center_y - y_size)
        bottom_y = min(height, center_y + y_size)

        return left_x, top_y, right_x, bottom_y
    
def process_frame(frame, tracking_data):
    """Process frame for respiration tracking"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    y_pos = None
    
    if len(tracking_data['features']) > 10:
        # Calculate optical flow
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            tracking_data['old_gray'], 
            frame_gray, 
            tracking_data['features'], 
            None,
            **tracking_data['lk_params']
        )
        
        if new_features is not None:
            # Select good points
            good_old = tracking_data['features'][status == 1]
            good_new = new_features[status == 1]
            
            # Calculate zone-based position
            y_pos = calculate_zone_based_position(good_new, tracking_data['roi_coords'])
            
            # Draw visualization
            frame = draw_zones_and_points(frame, tracking_data['roi_coords'], good_new)
            
            # Update features for next frame
            tracking_data['features'] = good_new.reshape(-1, 1, 2)
        
        # Update old frame
        tracking_data['old_gray'] = frame_gray.copy()
    else:
        # Reinitialize features if needed
        roi = frame_gray[tracking_data['roi_coords'][1]:tracking_data['roi_coords'][3],
                        tracking_data['roi_coords'][0]:tracking_data['roi_coords'][2]]
        new_features = cv2.goodFeaturesToTrack(roi, 
                                             maxCorners=60,
                                             qualityLevel=0.15,
                                             minDistance=3,
                                             blockSize=7)
        if new_features is not None:
            new_features = new_features + np.array(
                [[tracking_data['roi_coords'][0], tracking_data['roi_coords'][1]]], 
                dtype=np.float32
            )
            tracking_data['features'] = new_features
    
    return frame, y_pos

def calculate_zone_based_position(points, roi_coords):
    """Calculate weighted average y-position using zones"""
    left_x, top_y, right_x, bottom_y = roi_coords
    width = right_x - left_x
    height = bottom_y - top_y
    
    # Bagi ROI menjadi 3 zona vertikal
    zone_width = width // 3
    zones = {
        'left': [],
        'center': [],
        'right': []
    }
    
    # Assign points ke zona yang sesuai
    for point in points:
        x, y = point.ravel()
        relative_x = x - left_x
        
        if relative_x < zone_width:
            zones['left'].append(y)
        elif relative_x < 2 * zone_width:
            zones['center'].append(y)
        else:
            zones['right'].append(y)
    
    # Hitung weighted average dengan bobot yang lebih tinggi untuk zona tengah
    weights = {'left': 0.25, 'center': 0.5, 'right': 0.25}
    total_y = 0
    total_weight = 0
    
    for zone_name, points in zones.items():
        if points:  # Jika ada points di zona ini
            zone_avg = np.mean(points)
            weight = weights[zone_name] * len(points)
            total_y += zone_avg * weight
            total_weight += weight
    
    if total_weight > 0:
        return total_y / total_weight
    return None

def draw_zones_and_points(frame, roi_coords, good_new):
    """
    Menggambar zona dan titik-titik dengan warna berbeda
    """
    left_x, top_y, right_x, bottom_y = roi_coords
    roi_height = bottom_y - top_y
    zone_height = roi_height // 3

    # Buat overlay untuk zona dengan transparansi
    overlay = frame.copy()
    
    # Gambar zona dengan warna berbeda (BGR format)
    # Zona atas (merah muda transparan)
    cv2.rectangle(overlay, 
                 (left_x, top_y), 
                 (right_x, top_y + zone_height),
                 (147, 20, 255), -1)  # Pink
    
    # Zona tengah (hijau transparan)
    cv2.rectangle(overlay, 
                 (left_x, top_y + zone_height), 
                 (right_x, top_y + 2*zone_height),
                 (0, 255, 0), -1)  # Green
    
    # Zona bawah (biru transparan)
    cv2.rectangle(overlay, 
                 (left_x, top_y + 2*zone_height), 
                 (right_x, bottom_y),
                 (255, 191, 0), -1)  # Blue
    
    # Aplikasikan transparansi
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Gambar titik-titik dengan warna sesuai zonanya
    for point in good_new:
        x, y = point.ravel()
        relative_y = y - top_y
        
        if relative_y < zone_height:  # zona atas
            color = (147, 20, 255)  # Pink
        elif relative_y < 2 * zone_height:  # zona tengah
            color = (0, 255, 0)  # Green
        else:  # zona bawah
            color = (255, 191, 0)  # Blue
            
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    
    # Tambahkan label zona dan bobot
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Top Zone (0.2)", (right_x + 10, top_y + zone_height//2), 
                font, 0.5, (147, 20, 255), 2)
    cv2.putText(frame, "Middle Zone (0.5)", (right_x + 10, top_y + 3*zone_height//2), 
                font, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Bottom Zone (0.3)", (right_x + 10, top_y + 5*zone_height//2), 
                font, 0.5, (255, 191, 0), 2)
    
    return frame

def enhance_roi(roi):
    """Meningkatkan kualitas ROI untuk tracking yang lebih baik"""
    if roi is None or roi.size == 0:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Apply edge enhancement
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
    return enhanced