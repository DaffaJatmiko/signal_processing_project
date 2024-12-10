# utils.py:
import os
import platform
import requests
import subprocess
from tqdm import tqdm


def download_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return filename

    print(f"Downloading model to {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filename, "wb") as f, tqdm(total=int(response.headers.get("content-length", 0)), unit="iB", unit_scale=True) as pbar:
        for data in response.iter_content(1024):
            size = f.write(data)
            pbar.update(size)

    return filename


def check_gpu():
    system = platform.system()
    if system in ["Linux", "Windows"]:
        try:
            subprocess.check_output(["nvidia-smi"])
            return "NVIDIA"
        except Exception:
            return "CPU"
    elif system == "Darwin":
        try:
            cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            return "MLX" if "Apple" in cpu_info else "CPU"
        except Exception:
            pass
    return "CPU"