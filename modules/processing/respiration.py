# respiration.py
import time
import cv2
import numpy as np
import mediapipe as mp

def get_initial_roi(image, model_path, x_size, y_size, shift_x, shift_y):
    """
    Detects the initial Region of Interest (ROI) for respiration tracking using MediaPipe.

    Parameters:
    - image: The input image (BGR format).
    - model_path: Path to the MediaPipe model for pose detection.
    - x_size: Width of the ROI to be extracted.
    - y_size: Height of the ROI to be extracted.
    - shift_x: Horizontal shift to apply to the center of the ROI.
    - shift_y: Vertical shift to apply to the center of the ROI.

    Returns:
    - left_x: The left x-coordinate of the ROI.
    - top_y: The top y-coordinate of the ROI.
    - right_x: The right x-coordinate of the ROI.
    - bottom_y: The bottom y-coordinate of the ROI.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    detection_result = []
    
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

class FlowTracker:
    """
    Class to track flow positions based on detected points in the ROI.

    Attributes:
    - history_size: The number of previous positions to consider for smoothing.
    - last_positions: The last known positions of tracked points.
    - last_y_pos: The last calculated Y position for respiration.
    - smoothing_window: A list to store previous Y positions for smoothing.

    Methods:
    - get_flow_position(points, roi_coords): Calculates the smoothed Y position based on tracked points.
    """
    def __init__(self, history_size=8):  
        self.history_size = history_size
        self.last_positions = None
        self.last_y_pos = None
        self.smoothing_window = []
        
    def get_flow_position(self, points, roi_coords):
        """
        Calculates the flow position based on the tracked points within the ROI.

        Parameters:
        - points: List of detected points (x, y) within the ROI.
        - roi_coords: Coordinates of the ROI (left_x, top_y, right_x, bottom_y).

        Returns:
        - smoothed_y: The smoothed Y position based on valid points.
        """
        if len(points) == 0:
            return self.last_y_pos
            
        left_x, top_y, right_x, bottom_y = roi_coords
        width = right_x - left_x
        
        # Kembalikan bobot yang lebih seimbang
        zone_weights = {'left': 0.5, 'center': 1.0, 'right': 0.5}
        
        zones = {
            'left': (left_x, left_x + width/3),
            'center': (left_x + width/3, left_x + 2*width/3),
            'right': (left_x + 2*width/3, right_x)
        }
        
        zone_positions = {'left': [], 'center': [], 'right': []}
        
        for point in points:
            x, y = point.ravel()
            for zone_name, (zone_left, zone_right) in zones.items():
                if zone_left <= x <= zone_right:
                    zone_positions[zone_name].append(y)
                    break
        
        weighted_sum = 0
        total_weight = 0
        
        for zone_name, positions in zone_positions.items():
            if positions:
                zone_y = np.median(positions)
                weighted_sum += zone_y * zone_weights[zone_name]
                total_weight += zone_weights[zone_name]
        
        if total_weight > 0:
            y_pos = weighted_sum / total_weight
            
            self.smoothing_window.append(y_pos)
            if len(self.smoothing_window) > self.history_size:
                self.smoothing_window.pop(0)
            
            # Gunakan weighted average yang lebih ringan
            weights = np.linspace(0.5, 1.0, len(self.smoothing_window))
            weights /= weights.sum()
            smoothed_y = np.average(self.smoothing_window, weights=weights)
            
            self.last_y_pos = smoothed_y
            return smoothed_y
            
        return self.last_y_pos

def process_frame(frame, tracking_data):
    """
    Processes a single video frame for respiration tracking.

    Parameters:
    - frame: The current video frame (BGR format).
    - tracking_data: A dictionary containing tracking information, including:
        - 'flow_tracker': Instance of FlowTracker.
        - 'roi_coords': Coordinates of the ROI.
        - 'features': Detected features in the ROI.
        - 'old_gray': Previous grayscale frame for optical flow calculation.

    Returns:
    - frame: The processed frame with visual indicators.
    - y_pos: The calculated Y position for respiration.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    y_pos = None
    
    if 'flow_tracker' not in tracking_data:
        tracking_data['flow_tracker'] = FlowTracker()
    
    if len(tracking_data.get('features', [])) < 25:
        roi = frame_gray[tracking_data['roi_coords'][1]:tracking_data['roi_coords'][3],
                        tracking_data['roi_coords'][0]:tracking_data['roi_coords'][2]]
        
        # Enhanced pre-processing
        roi_enhanced = cv2.equalizeHist(roi)
        roi_enhanced = cv2.GaussianBlur(roi_enhanced, (5,5), 0)
        
        new_features = cv2.goodFeaturesToTrack(
            roi_enhanced, 
            maxCorners=50,  
            qualityLevel=0.15,  
            minDistance=7,  
            blockSize=7
        )
        
        if new_features is not None:
            new_features = new_features + np.array(
                [[tracking_data['roi_coords'][0], tracking_data['roi_coords'][1]]], 
                dtype=np.float32
            )
            tracking_data['features'] = new_features
    
    if len(tracking_data.get('features', [])) > 0:
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            tracking_data['old_gray'], 
            frame_gray, 
            tracking_data['features'], 
            None,
            winSize=(21,21), 
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if new_features is not None:
            good_new = new_features[status == 1]
            
            y_pos = tracking_data['flow_tracker'].get_flow_position(
                good_new, tracking_data['roi_coords']
            )
            
            frame = draw_flow_zones(frame, tracking_data['roi_coords'], good_new)
            tracking_data['features'] = good_new.reshape(-1, 1, 2)
    
    tracking_data['old_gray'] = frame_gray.copy()
    return frame, y_pos

def draw_flow_zones(frame, roi_coords, points):
    """
    Draws vertical zones and flow indicators on the frame.

    Parameters:
    - frame: The current video frame (BGR format).
    - roi_coords: Coordinates of the ROI (left_x, top_y, right_x, bottom_y).
    - points: List of tracked points to visualize.

    Returns:
    - frame: The frame with drawn zones and points.
    """
    left_x, top_y, right_x, bottom_y = roi_coords
    width = right_x - left_x
    
    # Draw main ROI
    cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 255, 0), 2)
    
    # Draw vertical zones
    for i in range(1, 3):
        x = left_x + (width * i // 3)
        cv2.line(frame, (x, top_y), (x, bottom_y), (0, 255, 255), 1)
    
    # Draw points with zone-based colors
    if points is not None:
        for point in points:
            x, y = point.ravel()
            x, y = int(x), int(y)
            
            # Determine zone
            rel_x = x - left_x
            if rel_x < width/3:
                color = (0, 165, 255)  # Orange for left
            elif rel_x < 2*width/3:
                color = (0, 0, 255)    # Red for center
            else:
                color = (0, 165, 255)  # Orange for right
                
            cv2.circle(frame, (x, y), 3, color, -1)
    
    return frame

def enhance_roi(roi):
    """
    Enhances the quality of the ROI for better tracking.

    Parameters:
    - roi: The Region of Interest (ROI) image (BGR format).

    Returns:
    - enhanced: The enhanced ROI image (grayscale).
    """
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
