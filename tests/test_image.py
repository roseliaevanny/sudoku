import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.image_processing import (
    pre_process_image,
    find_sudoku_contour,
    reorder_points,
    warp_perspective,
    split_boxes,
    clean_and_prepare_box,
    _estimate_empty_threshold,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

IMG_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'sudoku.png')

# Load
img_bgr = cv2.imread(IMG_PATH)
assert img_bgr is not None, f"Could not load image: {IMG_PATH}"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print(f"Loaded image: {img_bgr.shape[1]}×{img_bgr.shape[0]} px")

# Stage 1: pre_process_image
thresh = pre_process_image(img_bgr)
print(f"Stage 1 – pre_process_image: output shape {thresh.shape}, dtype {thresh.dtype}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis('off')
axes[1].imshow(thresh, cmap='gray'); axes[1].set_title("Stage 1: Threshold (BINARY_INV)"); axes[1].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '1_threshold.png'), dpi=120)
plt.close(fig)

# Stage 2: find_sudoku_contour
try:
    corners = find_sudoku_contour(thresh)
    print(f"Stage 2 – find_sudoku_contour: found 4 corners\n  {corners.reshape(4,2).tolist()}")
except ValueError as e:
    print(f"Stage 2 FAILED: {e}")
    sys.exit(1)

# Draw corners on original image.
# reorder_points is called first so labels match the actual warp order.
vis = img_rgb.copy()
ordered_corners = reorder_points(corners)
pts_raw = corners.reshape(4, 2).astype(int)
pts_ordered = ordered_corners.reshape(4, 2).astype(int)

# Draw the contour outline (using raw order for the polygon)
cv2.polylines(vis, [pts_raw], isClosed=True, color=(0, 255, 0), thickness=3)

# Label using reordered corners so TL/TR/BR/BL are correct
for i, pt in enumerate(pts_ordered):
    labels = ['TL', 'TR', 'BR', 'BL']
    color = [(255,80,80), (80,255,80), (255,200,0), (80,180,255)][i]
    cv2.circle(vis, tuple(pt), 10, color, -1)
    cv2.putText(vis, labels[i], tuple(pt + np.array([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis('off')
axes[1].imshow(vis); axes[1].set_title("Stage 2: Detected Grid Contour + Corners"); axes[1].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '2_contour.png'), dpi=120)
plt.close(fig)

# Stage 3: warp_perspective
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
warped_gray = warp_perspective(gray, corners, size=450)
print(f"Stage 3 – warp_perspective: output shape {warped_gray.shape}")

# Re-binarise the warped image for cell processing
warped_blur = cv2.GaussianBlur(warped_gray, (3, 3), 0)
warped_thresh = cv2.adaptiveThreshold(
    warped_blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis('off')
axes[1].imshow(warped_thresh, cmap='gray'); axes[1].set_title("Stage 3: Warped + Re-thresholded"); axes[1].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '3_warped.png'), dpi=120)
plt.close(fig)

# Stage 4: split_boxes
boxes = split_boxes(warped_thresh)
print(f"Stage 4 – split_boxes: {len(boxes)} cells, each {boxes[0].shape}")

fig, axes = plt.subplots(9, 9, figsize=(9, 9))
for i, (ax, box) in enumerate(zip(axes.flat, boxes)):
    ax.imshow(box, cmap='gray')
    ax.axis('off')
fig.suptitle("Stage 4: 81 Split Cells", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '4_split_cells.png'), dpi=120, bbox_inches='tight')
plt.close(fig)

# Stage 5: clean_and_prepare_box
auto_thresh = _estimate_empty_threshold(boxes)
print(f"Stage 5 – auto empty threshold: {auto_thresh:.4f}")
cleaned = [clean_and_prepare_box(b, empty_threshold=auto_thresh) for b in boxes]
non_empty = [(i, c) for i, c in enumerate(cleaned) if c is not None]
empty_count = sum(1 for c in cleaned if c is None)
print(f"Stage 5 – clean_and_prepare_box: {len(non_empty)} digits found, {empty_count} empty cells")

fig, axes = plt.subplots(9, 9, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    cell = cleaned[i]
    if cell is not None:
        ax.imshow(cell, cmap='gray')
        ax.set_facecolor('#1a1a2e')
    else:
        ax.set_facecolor('#2d2d44')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#555577')
        spine.set_linewidth(0.5)

fig.suptitle(f"Stage 5: Prepared Cells (28×28)\n{len(non_empty)} digits | {empty_count} empty",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '5_prepared_cells.png'), dpi=120, bbox_inches='tight')
plt.close(fig)

# Full pipeline summary
print("\nPipeline Summary")
grid_labels = []
for i, cell in enumerate(cleaned):
    grid_labels.append("D" if cell is not None else ".")

print("Grid (D = digit, . = empty):")
for row in range(9):
    row_labels = grid_labels[row*9:(row+1)*9]
    digit_count = row_labels.count("D")
    print(f"  Row {row+1}: {' '.join(row_labels)}  ({digit_count} digits)")

# Composite summary figure
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0f0f1a')

stages = [
    (img_rgb,       "Original"),
    (thresh,        "1. Threshold"),
    (vis,           "2. Grid Contour"),
    (warped_thresh, "3. Warped"),
]
cmaps = [None, 'gray', None, 'gray']

for idx, ((img, title), cmap) in enumerate(zip(stages, cmaps)):
    ax = fig.add_subplot(2, 4, idx + 1)
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, color='white', fontsize=10, pad=6)
    ax.axis('off')

# Show a 3×3 sample of cleaned cells
sample_indices = [i for i, c in enumerate(cleaned) if c is not None][:9]
for plot_pos, cell_idx in enumerate(sample_indices):
    ax = fig.add_subplot(2, 9, 10 + plot_pos)
    ax.imshow(cleaned[cell_idx], cmap='gray')
    ax.set_title(f"#{cell_idx}", color='#aaaacc', fontsize=7)
    ax.axis('off')

fig.suptitle("image_processing.py — Full Pipeline Results", color='white', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'summary.png'), dpi=130, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\nAll output images saved to: {os.path.abspath(OUT_DIR)}")