import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

# CONFIGURATION

# input_dir = r"data\Trial 1 (water on glass)\Static"
root = tk.Tk()
root.withdraw()  # Hide the root window
root.attributes('-topmost', True)  # Keep the file dialog on top
input_dir = filedialog.askdirectory()

output_dir = os.path.join(input_dir, "masks_json")
os.makedirs(output_dir, exist_ok=True)

# checkpoint_path = '/home/u15/sraglow/Documents/Github/segmenteverygrain/segmenteverygrain/sam_vit_h_4b8939.
checkpoint_path = r"C:\Users\Aj\Documents\GitHub\segmenteverygrain\seg_tf_workspace\sam_vit_h_4b8939.pth"
model_type = "vit_h"

MAX_DIM = 1024
RELEVANT_AREA_MIN = 5000
CENTER_THRESHOLD = 100
DISPLAY_EVERY_N = 1

# Load SAM model
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

# Helper: Visualization
def show_masks_on_image(image, masks, alpha=0.4):
    image = image.copy()
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    cmap = cm.get_cmap('viridis', len(masks))

    for i, m in enumerate(masks):
        rgba = cmap(i)
        color = (np.array(rgba[:3]) * 255).astype(np.uint8)
        binary_mask = m["segmentation"]
        mask_overlay[binary_mask] = color

        y_indices, x_indices = np.where(binary_mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            cx, cy = int(np.mean(x_indices)), int(np.mean(y_indices))
            cv2.putText(image, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    blended = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Segmented Masks")
    plt.savefig(f"{output_dir}/preview_{path.stem}.png", bbox_inches='tight')
    plt.close()

# Process all .tif images
image_paths = sorted(Path(input_dir).glob("*.tif"))  + sorted(Path(input_dir).glob("*.png")) + sorted(Path(input_dir).glob("*.jpg"))
print(f"Found {len(image_paths)} images in {input_dir}")
for idx, path in enumerate(tqdm(image_paths, desc = 'processing images')):
    out_json = Path(output_dir) / f"{path.stem}_masks.json"
    if out_json.exists():
        print(f"Skipping {path.name} (already processed)")
        continue
    print(f"Processing {path.name} ({idx+1}/{len(image_paths)})")
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    masks = mask_generator.generate(image)
    print(f"Generated {len(masks)} masks")

    relevant_masks = [m for m in masks if m['area'] > RELEVANT_AREA_MIN]
    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    relevant_masks_center = [m for m in relevant_masks if abs(m['bbox'][0] + m['bbox'][2] / 2 - cx) < CENTER_THRESHOLD]

    
    with open(out_json, "w") as f:
        json.dump(masks, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    if idx % DISPLAY_EVERY_N == 0:
        show_masks_on_image(image, relevant_masks)
