import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURATION ---
checkpoint_path   = r"C:\Users\Aj\Documents\GitHub\segmenteverygrain\seg_tf_workspace\sam_vit_h_4b8939.pth"
model_type        = "vit_h"
MAX_DIM           = 1024
area_threshold    = 1000    # ← only keep masks with area ≥ this
CENTER_THRESHOLD  = 100
DISPLAY_EVERY_N   = 1

# Load SAM model once
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

def show_masks_on_image(image, masks, out_png, alpha=0.4):
    img     = image.copy()
    overlay = np.zeros_like(img, dtype=np.uint8)
    cmap    = cm.get_cmap('viridis', len(masks))

    for i, m in enumerate(masks):
        color = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        bm    = m["segmentation"]
        overlay[bm] = color

        ys, xs = np.where(bm)
        if len(xs) and len(ys):
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(img, str(i), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    blended = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    plt.figure(figsize=(10,10))
    plt.imshow(blended); plt.axis("off")
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

# --- GUI: pick multiple folders ---
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

input_dirs = []
print("Select folders to process (hit Cancel when done)...")
while True:
    d = filedialog.askdirectory(title="Select folder (Cancel to finish)")
    if not d:
        break
    input_dirs.append(d)

if not input_dirs:
    print("No folders selected, exiting.")
    sys.exit(0)

# --- Process each folder ---
for input_dir in input_dirs:
    print(f"\n=== Processing folder: {input_dir} ===")
    output_dir = os.path.join(input_dir, "masks_json")
    os.makedirs(output_dir, exist_ok=True)

    p = Path(input_dir)
    image_paths = sorted(p.glob("*.tif")) + sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
    print(f" Found {len(image_paths)} images")

    for idx, path in enumerate(tqdm(image_paths, desc="Images")):
        out_json = Path(output_dir)/f"{path.stem}_masks.json"
        if out_json.exists():
            continue

        # load & resize
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = MAX_DIM/max(h, w)
        if scale < 1:
            img = cv2.resize(img,
                             (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_AREA)

        # segment
        masks = mask_generator.generate(img)

        # --- FILTER BY AREA BEFORE SAVING ---
        filtered = [m for m in masks if m['area'] >= area_threshold]
        print(f"  → {path.name}: kept {len(filtered)}/{len(masks)} masks")

        # save filtered JSON
        with open(out_json, "w") as f:
            json.dump(filtered, f, indent=2,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        # optional preview of only the large & centered ones
        if idx % DISPLAY_EVERY_N == 0 and filtered:
            cx, cy = img.shape[1]//2, img.shape[0]//2
            centered = [
                m for m in filtered
                if abs(m['bbox'][0] + m['bbox'][2]/2 - cx) < CENTER_THRESHOLD
            ]
            preview_png = Path(output_dir)/f"preview_{path.stem}.png"
            show_masks_on_image(img, centered, str(preview_png))

    print(f"→ Done with {input_dir}, results in {output_dir}")
