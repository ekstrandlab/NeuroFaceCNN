#!/usr/bin/env python3

"""
Moment-by-moment fMRI face prediction using CNN classifier
"""

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score


MODEL_FILE = r"D:\Face-Project\best-result-cnn\model85-new.h5"
FMRI_FILE  = r"C:\sub-1_task-500daysofsummer_bold_preprocessedICA.nii.gz"
MASK_PATH  = r"C:\85_subBrainMask_average_99.nii.gz"
EVENTS_FILE = r"C:\task-500daysofsummer_face-annotation.1D.txt"

OUT_DIR = Path(r"C:\Users\sara.asadi\predictions")
OUT_DIR.mkdir(exist_ok=True)

# fMRI + CNN params
TR = 1.0
LAG = 4
T_START = 2032
T_END   = 3032
WIN = 10
STEP = 1

# LOAD GROUND TRUTH 
print("Loading ground truth events...")
events_data = np.loadtxt(EVENTS_FILE)
gt_onsets_unshifted = events_data[:, 0]
gt_durations        = events_data[:, 1]
print(f"Loaded {len(gt_onsets_unshifted)} events")


# Mark the event window as 1, rest as 0
def create_ground_truth_timeline(onsets, durations, start_time, end_time, resolution=1.0):
    """Return binary ground truth timeline over a time axis."""
    t_axis = np.arange(start_time, end_time, resolution)
    gt = np.zeros(len(t_axis))

    for onset, duration in zip(onsets, durations):
        end = onset + duration
        idx = np.where((t_axis >= onset) & (t_axis < end))[0]
        gt[idx] = 1

    return t_axis, gt


# LOAD MODEL / MASK / fMRI  
print("Loading model...")
model = tf.keras.models.load_model(MODEL_FILE)

print("Loading mask...")
mask_img = nib.load(MASK_PATH).get_fdata()
mask_idx = np.where(mask_img.reshape(-1) > 0.99)[0]
print("Mask voxels kept:", len(mask_idx))

print("Loading fMRI...")
fmri = nib.load(FMRI_FILE).get_fdata()
X, Y, Z, T = fmri.shape

flat = fmri.reshape(-1, T)
masked_full = flat[mask_idx, :]
np.save(OUT_DIR / "full_vox_by_time.npy", masked_full)


# EXTRACT LAGGED SEGMENT  
stim_start_vol = int(T_START / TR)
stim_end_vol   = int(T_END   / TR)

bold_start = stim_start_vol + LAG
bold_end   = stim_end_vol   + LAG

segment = masked_full[:, bold_start:bold_end]
print("Segment:", segment.shape)
np.save(OUT_DIR / "segment.npy", segment)


# SLIDING WINDOW  
windows = []
times = []

# Slide a window of length WIN across the time dimension, with step size STEP=1
# Segment.shape[1] = number of timepoints
for t in range(0, segment.shape[1] - WIN + 1, STEP):
    win = segment[:, t:t+WIN]
    windows.append(win)

    win_start_bold = bold_start + t                # t -> the start index of the sliding window
    win_start_time = win_start_bold * TR
    times.append(win_start_time)

windows = np.array(windows, dtype=np.float32)
times   = np.array(times, dtype=float)

print("Windows:", windows.shape)
print("Time range:", times.min(), "→", times.max())

# CNN expects channel dimension
windows = windows[..., np.newaxis]   # (N, vox, 10, 1)


# PREDICT  
preds = model.predict(windows, batch_size=16).reshape(-1)
binary_preds = (preds >= 0.5).astype(int)

np.save(OUT_DIR / "predictions.npy", preds)
np.save(OUT_DIR / "predictions_binary.npy", binary_preds)


#  GROUND TRUTH ALIGNMENT 
gt_time, gt_binary = create_ground_truth_timeline(
    gt_onsets_unshifted, gt_durations, times.min(), times.max())

# Align GT to prediction sample times
gt_aligned = np.zeros(len(times))
for i, t in enumerate(times):
    idx = np.argmin(np.abs(gt_time - t))
    gt_aligned[i] = gt_binary[idx]


# PERFORMANCE METRICS  
precision = precision_score(gt_aligned, binary_preds)
recall = recall_score(gt_aligned, binary_preds)
f1 = f1_score(gt_aligned, binary_preds)

print("\n METRICS: ")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# PROBABILITY PLOT
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(times, preds, color='blue', label='CNN Probability')
ax.fill_between(gt_time, 0, gt_binary, color='green', alpha=0.3, label='GT (0s)')
ax.axhline(0.5, linestyle='--', color='red', alpha=0.5)
ax.set_title("Predicted Probability vs Ground Truth")
ax.set_xlabel("Time (s, lagged BOLD time)")
ax.set_ylabel("P(face)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_probability.png")
plt.close()


# BINARY PLOT
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(times, binary_preds, drawstyle='steps-post',
        color='blue', linewidth=1.5, label='CNN Binary')
ax.fill_between(gt_time, 0, gt_binary,
                color='green', alpha=0.3, label='GT (0s)')
ax.set_title("Binary Predictions vs Ground Truth")
ax.set_xlabel("Time (s, lagged BOLD time)")
ax.set_ylabel("Prediction (0 or 1)")
ax.set_ylim(-0.1, 1.1)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_binary.png")
plt.close()

print("\n All results saved to:", OUT_DIR)

