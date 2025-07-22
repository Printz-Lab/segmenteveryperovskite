#%%
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np

# Load the SAM model (adjust to sam2 if needed)
# checkpoint_path = r"C:\Users\Aj\Documents\GitHub\segmenteverygrain\seg_tf_workspace\sam_vit_h_4b8939.pth"
checkpoint_path = r"C:\Users\Aj\Documents\GitHub\segmenteverygrain\seg_tf_workspace\sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to("cuda")

#%%
# mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=20,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.9,
    min_mask_region_area=25000,
    crop_n_layers=0,
)



# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.85,
#     stability_score_thresh=0.9,
#     min_mask_region_area=100,
#     crop_n_layers=0,  # Reduce GPU load
# )
filename = r"data\Trial 1 (water on glass)\Static\Control_100um_MQ.tif"

image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
import cv2

import torch
#%%

# Estimate how many masks will be generated
points_per_side = 40  # Match this to your actual mask_generator config
num_points = points_per_side ** 2

h, w = image.shape[:2]
estimated_elements = num_points * h * w
limit = torch.iinfo(torch.int32).max

print(f"Estimated total tensor elements: {estimated_elements:,}")
print(f"Limit (INT_MAX): {limit:,}")
print(f"Ratio used: {estimated_elements / limit:.2%}")

MAX_DIM = 4096  # or 768, depending on your GPU

# Resize image if too large
h, w = image.shape[:2]
scale = MAX_DIM / max(h, w)
if scale < 1:
    image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
h, w = image.shape[:2]
print(f"Resized image to {w}x{h} pixels.")
estimated_elements = num_points * h * w
print(f"Estimated total tensor elements after resize: {estimated_elements:,}")
print(f"Ratio used after resize: {estimated_elements / limit:.2%}")


#%%
import time 

start_time = time.time()

masks = mask_generator.generate(image)
print(f"Generated {len(masks)} masks.")

end_time = time.time()

print(f"Time taken to generate masks: {end_time - start_time:.2f} seconds.")
relevant_masks = [m for m in masks if 5000 < m['area']]
print(len(relevant_masks))
# relevant_masks2 = [m for m in relevant_masks if m['stability_score']>0.9]
# cx, cy = image.shape[1] // 2, image.shape[0] // 2
# relevant_masks3 = [m for m in relevant_masks2 if abs(m['bbox'][1] + m['bbox'][3] / 2 - cy) < 100]
# print(len(relevant_masks3))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# Visualize the original image with all masks overlaid
def show_masks_on_image(image, masks, alpha=0.5):
    print(f"Showing {len(masks)} masks on image.")
    if isinstance(image, str):  # handle if path passed
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image.copy()

    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    cmap = cm.get_cmap('viridis', len(masks))  # or 'hsv', 'jet', 'viridis'

    # Assign each mask a color
    for i, m in enumerate(masks):
        rgba = cmap(i)  # returns (r, g, b, a) with float values 0–1
        color = (np.array(rgba[:3]) * 255).astype(np.uint8)  # convert to [0, 255]
        binary_mask = m["segmentation"]
        mask_overlay[binary_mask] = color

            # Apply color to mask region
        mask_overlay[binary_mask] = color

        # Find mask centroid for label
        y_indices, x_indices = np.where(binary_mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))
            cv2.putText(
                image,
                str(i),
                (center_x + 10*i, center_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

    # Blend original and mask overlay
    blended = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.title("Segmented Masks over Original Image")
    plt.axis("off")
    plt.show()

import json
import os


file = os.path.basename(filename)

with open(f"{file}_masks_metadata.json", "w") as f:
    print(f"Saving masks metadata to {f.name}")
    json.dump(masks, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


show_masks_on_image(image, masks)
show_masks_on_image(image, relevant_masks)
# show_masks_on_image(image, relevant_masks2)
# show_masks_on_image(image, relevant_masks3)
# %%
import torch
total = torch.cuda.get_device_properties(0).total_memory
reserved = torch.cuda.memory_reserved(0)
allocated = torch.cuda.memory_allocated(0)

print(f"Total     : {total / 1024**2:.2f} MB")
print(f"Reserved  : {reserved / 1024**2:.2f} MB")
print(f"Allocated : {allocated / 1024**2:.2f} MB")
print(f"Free      : {(reserved - allocated) / 1024**2:.2f} MB (within reserved)")

# %%
