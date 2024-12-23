# This is temporary code for rPPG
# To run rPPG use py mainrppg.py
# py main.py will run respiration instead rPPG

import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
import time

def cpu_POS(signal, fps):
    eps = 10**-9
    X = signal
    e, c, f = X.shape
    w = int(1.6 * fps)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(M, Cn)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)
    return H

def process_rppg_signals(r_signal, g_signal, b_signal, fps):
    if len(r_signal) > 0:
        rgb_signals = np.array([r_signal, g_signal, b_signal])
        rgb_signals = rgb_signals.reshape(1, 3, -1)
        rppg_signal = cpu_POS(rgb_signals, fps=fps).reshape(-1)

        # Chebyshev filter
        fs = int(fps)
        lowcut, highcut = 0.9, 2.4

        # Design a Chebyshev Type I filter
        ripple = 0.5  # Passband ripple in dB
        b, a = signal.cheby1(N=3, rp=ripple, Wn=[lowcut, highcut], btype='band', fs=fs)
        
        # Apply Filters
        filtered_rppg = signal.filtfilt(b, a, rppg_signal)

        # Heart rate calculation
        prominence = 0.5 * np.std(filtered_rppg)
        peaks, _ = signal.find_peaks(filtered_rppg, prominence=prominence)
        heart_rate = 60 * len(peaks) / (len(filtered_rppg) / fs)

        print(f"Heart Rate: {heart_rate:.2f} BPM")

        return filtered_rppg, heart_rate
    else:
        print("No rPPG signal extracted. Please ensure a face is visible in the webcam feed.")
        return None, None

def main():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # Change video source to webcam (use 0 for default webcam)
    cap = cv2.VideoCapture(0)

    # Set webcam FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Check if the webcam is accessible
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    fps = 30  # Fixed FPS
    r_signal, g_signal, b_signal = [], [], []
    start_time = time.time()
    timeout = 60  # Timeout after 1 minute
    frame_count = 0
    all_rppg_results = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam. Exiting...")
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            # Detect face and extract ROI
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    # Bounding box for the face (relative to the full frame)
                    face_x1 = int(bboxC.xmin * w)
                    face_y1 = int(bboxC.ymin * h)
                    face_x2 = int((bboxC.xmin + bboxC.width) * w)
                    face_y2 = int((bboxC.ymin + bboxC.height) * h)

                    # Face dimensions
                    face_width = face_x2 - face_x1
                    face_height = face_y2 - face_y1
                    
                    # ROI Dimensions based on proportions
                    roi_width = int(0.4 * face_width)  # 40% of face width
                    roi_height = int(0.3 * face_height)  # 30% of face height
                    
                    # ROI Position (10% above face bounding box and centered)
                    roi_x1 = face_x1 + int(0.3 * face_width)  # 30% offset from the left
                    roi_y1 = face_y1 - int(0.1 * face_height)  # 10% above the top of the bounding box
                    roi_x2 = roi_x1 + roi_width
                    roi_y2 = roi_y1 + roi_height

                    # Ensure ROI stays within frame boundaries
                    roi_x1 = max(0, roi_x1)
                    roi_y1 = max(0, roi_y1)
                    roi_x2 = min(w, roi_x2)
                    roi_y2 = min(h, roi_y2)

                    # Extract and process ROI
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi.size > 0:
                        r_signal.append(np.mean(roi[:, :, 0]))
                        g_signal.append(np.mean(roi[:, :, 1]))
                        b_signal.append(np.mean(roi[:, :, 2]))

                    # Draw the bounding box around the ROI
                    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

                    # Draw the face bounding box (optional for visualization)
                    cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)

            # Display the frame with bounding box
            cv2.imshow('Webcam Feed', frame)

            frame_count += 1

            # Process and print rPPG every 300 frames
            if frame_count % 300 == 0:
                print(f"Processing rPPG for frame batch {frame_count // 300}")
                filtered_rppg, heart_rate = process_rppg_signals(r_signal, g_signal, b_signal, fps)
                if filtered_rppg is not None:
                    all_rppg_results.extend(filtered_rppg)
                r_signal, g_signal, b_signal = [], [], []

            # Break the loop after timeout
            if (time.time() - start_time) > timeout:
                break

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Plot the combined rPPG result
        if all_rppg_results:
            fs = int(fps)
            time_axis = np.linspace(0, len(all_rppg_results) / fs, len(all_rppg_results))
            
            prominance = .5 * np.std(all_rppg_results)
            peaks, _ = signal.find_peaks(all_rppg_results, prominence=prominance)

            duration = len(all_rppg_results) / fs
            heart_rate = 60 * len(peaks) / duration

            print(f"Combined Heart Rate: {heart_rate:.2f} BPM")

            plt.figure(figsize=(20, 5))
            plt.plot(time_axis, all_rppg_results, color='black', label='rPPG Signal')
            plt.plot(np.array(peaks) / fs, np.array(all_rppg_results)[peaks], 'r.', label='Detected Peaks')
            plt.title(f'Combined rPPG Signal\nHeart Rate: {heart_rate:.2f} BPM')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
