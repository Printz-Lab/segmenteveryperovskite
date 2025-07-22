import cv2
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def extract_frames(video_path, output_folder, image_format='png'):
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert from BGR to RGB if needed (you often use RGB later)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_filename = output_folder / f"frame_{frame_idx:04d}.{image_format}"
        cv2.imwrite(str(frame_filename), frame)
        frame_idx += 1

    cap.release()
    print(f"Finished extracting {frame_idx} frames to {output_folder}")


if __name__ == "__main__":
    # --- Example usage ---
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Keep the file dialog on top

    video_file = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    output_dir = filedialog.askdirectory(
        title="Select Output Directory"
    )
    extract_frames(video_file, output_dir)