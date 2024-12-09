from modules.input.webcam import capture_webcam
from modules.processing.rppg import extract_rppg
from modules.processing.respiration import extract_respiration_signal
from modules.visualization.plotter import plot_signal


def main():
    print("Starting signal extraction...")

    # Capture webcam frames
    frames = capture_webcam(frame_count=300)

    # Extract rPPG signal
    rppg_signal = extract_rppg(frames)
    print("rPPG Signal Extracted")

    # Extract respiration signal
    respiration_signal = extract_respiration_signal(frames)
    print("Respiration Signal Extracted")

    # Visualize rPPG signal
    plot_signal(rppg_signal, title="rPPG Signal", xlabel="Time (frames)", ylabel="Amplitude")

    # Visualize respiration signal
    plot_signal(respiration_signal, title="Respiration Signal", xlabel="Time (frames)", ylabel="Amplitude")


if __name__ == "__main__":
    main()
