# This is temporary code for rPPG
# To run rPPG use py mainrppg.py
# py main.py will run respiration instead rPPG

import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal

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

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Change video source to webcam (use 0 for default webcam)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is accessible
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Use default FPS if not available
    r_signal, g_signal, b_signal = [], [], []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam. Exiting...")
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            # Detect face mesh and extract ROI
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Use the region around the nose tip (landmark 1) for ROI
                    nose_tip = face_landmarks.landmark[1]
                    h, w, _ = frame.shape
                    cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)

                    # Define a bounding box around the nose tip
                    bbox_size = 70  # Adjust the size as needed
                    x1, y1 = max(0, cx - bbox_size), max(0, cy - bbox_size)
                    x2, y2 = min(w, cx + bbox_size), min(h, cy + bbox_size)

                    # Extract and process ROI
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        r_signal.append(np.mean(roi[:, :, 0]))
                        g_signal.append(np.mean(roi[:, :, 1]))
                        b_signal.append(np.mean(roi[:, :, 2]))

                    # Draw the bounding box around the ROI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw the facial landmarks (optional)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                    )

            # Display the frame with bounding box and landmarks
            cv2.imshow('Webcam Feed', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Error: {e}")

    # Process rPPG Signal
    if len(r_signal) > 0:
        rgb_signals = np.array([r_signal, g_signal, b_signal])
        rgb_signals = rgb_signals.reshape(1, 3, -1)
        rppg_signal = cpu_POS(rgb_signals, fps=fps).reshape(-1)

        # Bandpass filter
        fs = int(fps)
        lowcut, highcut = 0.9, 2.4
        b, a = signal.butter(3, [lowcut, highcut], btype='band', fs=fs)
        filtered_rppg = signal.filtfilt(b, a, rppg_signal)

        # Heart rate calculation
        prominence = 0.5 * np.std(filtered_rppg)
        peaks, _ = signal.find_peaks(filtered_rppg, prominence=prominence)
        heart_rate = 60 * len(peaks) / (len(filtered_rppg) / fs)

        print(f"Heart Rate: {heart_rate:.2f} BPM")

        # Visualization
        plt.figure(figsize=(20, 5))
        plt.plot(filtered_rppg, color='black')
        plt.plot(peaks, filtered_rppg[peaks], 'x', color='red')
        plt.title(f'Heart Rate: {heart_rate:.2f} BPM')
        plt.tight_layout()
        plt.show()
    else:
        print("No rPPG signal extracted. Please ensure a face is visible in the webcam feed.")

if __name__ == '__main__':
    main()
