import matplotlib.pyplot as plt
import cv2
import numpy as np

from modules.processing import filters

def overlay_plot_on_frame(frame, timestamps, y_positions):
    if not timestamps or not y_positions:  # Skip if no data
        return frame

    # Use Agg backend which doesn't require a GUI
    plt.switch_backend('Agg')
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(timestamps, y_positions, "g-", linewidth=2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("Shoulder Movement")

    # Convert plot to image
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Clean up
    plt.close(fig)
    
    # Resize plot_img to fit in a corner of the frame
    h, w = frame.shape[:2]
    plot_h, plot_w = plot_img.shape[:2]
    scale = min(h/3/plot_h, w/3/plot_w)  # Make plot 1/3 of frame size
    new_h, new_w = int(plot_h*scale), int(plot_w*scale)
    plot_img = cv2.resize(plot_img, (new_w, new_h))
    
    # Overlay plot in top-right corner
    frame[10:10+new_h, w-new_w-10:w-10] = plot_img
    
    return frame

def plot_shoulder_movement(timestamps, y_positions):
    """Plot hasil akhir gerakan bahu dengan filtering"""
    if len(timestamps) > 0 and len(y_positions) > 0:
        print(f"Plotting data with {len(timestamps)} points...")
        
        # Switch back to default backend
        plt.switch_backend('TkAgg')
        
        plt.figure(figsize=(12, 6))
        # Normalisasi waktu ke detik
        t = np.array(timestamps) - timestamps[0]
        
        # Pre-process signal
        y_preprocessed = filters.preprocess_signal(y_positions)
        
        # Adjust timestamps for preprocessed signal
        t_preprocessed = t[:len(y_preprocessed)]
        
        # Normalisasi ke nilai mean=0
        y_normalized = y_preprocessed - np.mean(y_preprocessed)
        
        # Aplikasikan bandpass filter
        y_filtered = filters.apply_bandpass_filter(y_normalized)
        
        # Estimasi respiratory rate
        breaths_per_minute, peaks = filters.estimate_respiratory_rate(y_filtered, t_preprocessed)
        
        # Plot signals
        plt.plot(t_preprocessed, y_normalized, label='Preprocessed Signal', color='blue', alpha=0.3)
        plt.plot(t_preprocessed, y_filtered, label='Filtered Signal', color='red')
        plt.plot(t_preprocessed[peaks], y_filtered[peaks], "x", label='Peaks')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Y Position (pixels)')
        plt.title(f'Shoulder Movement Over Time\nEstimated Respiratory Rate: {breaths_per_minute:.1f} breaths/min')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"Estimated Respiratory Rate: {breaths_per_minute:.1f} breaths/min")
        
    return timestamps, y_positions