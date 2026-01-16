#!/usr/bin/env python3
"""
Moment-by-moment fMRI face prediction using CNN classifier
"""

import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score


# CONFIG
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_ROOT / "Output" / "CNN-result" / "model83.h5"
MASK_PATH = PROJECT_ROOT / "Output" / "2D-matrices" / "85_subBrainMask_average.nii.gz"

# Choose a subject + task
SUBJECT = "sub-01"
TASK = "500daysofsummer"

FMRI_FILE = (
    PROJECT_ROOT / "NNDB_ROOT" / "all-subjects" / SUBJECT / "func" /
    f"{SUBJECT}_task-{TASK}_bold_preprocessedICA.nii.gz")

EVENTS_FILE = (
    PROJECT_ROOT / "NNDB_ROOT" / "stimuli" /
    f"stimuli-task-{TASK}_face-annotation.1D")

OUT_DIR = PROJECT_ROOT / "Output" / "moment-by-moment-result" / SUBJECT / f"task-{TASK}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fmri = nib.load(str(FMRI_FILE)).get_fdata()
X, Y, Z, T = fmri.shape
print(f"fMRI shape: {(X, Y, Z, T)}")

# full movie duration
T_START = 0
T_END = T * TR


# params
TR = 1.0
LAG = 4
WIN = 10
STEP = 1


# LOAD GROUND TRUTH
print("Loading ground truth events...")
if not EVENTS_FILE.exists():
    raise FileNotFoundError(f"Events file not found: {EVENTS_FILE}")

events_data = np.loadtxt(str(EVENTS_FILE))
gt_onsets_unshifted = events_data[:, 0]
gt_durations        = events_data[:, 1]
print(f"Loaded {len(gt_onsets_unshifted)} events")

def create_ground_truth_timeline(onsets, durations, start_time, end_time, resolution=1.0):
    """Return binary ground truth timeline over a time axis."""
    t_axis = np.arange(start_time, end_time, resolution)
    gt = np.zeros(len(t_axis), dtype=np.float32)

    for onset, duration in zip(onsets, durations):
        end = onset + duration
        idx = np.where((t_axis >= onset) & (t_axis < end))[0]
        gt[idx] = 1

    return t_axis, gt


# LOAD MODEL / MASK / fMRI
print("Loading model...")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
model = tf.keras.models.load_model(str(MODEL_FILE), compile=False)

print("Loading mask...")
if not MASK_PATH.exists():
    raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

mask_img = nib.load(str(MASK_PATH)).get_fdata()
mask_idx = np.where(mask_img.reshape(-1) > 0.99)[0]
print("Mask voxels kept:", len(mask_idx))

print("Loading fMRI...")
if not FMRI_FILE.exists():
    raise FileNotFoundError(f"fMRI file not found: {FMRI_FILE}")

fmri = nib.load(str(FMRI_FILE)).get_fdata()
X, Y, Z, T = fmri.shape
print(f"fMRI shape: {(X, Y, Z, T)}")

flat = fmri.reshape(-1, T)
masked_full = flat[mask_idx, :]
np.save(str(OUT_DIR / "full_vox_by_time.npy"), masked_full)


# EXTRACT LAGGED SEGMENT
stim_start_vol = int(T_START / TR)
stim_end_vol   = int(T_END   / TR)

bold_start = stim_start_vol + LAG
bold_end   = stim_end_vol   + LAG

segment = masked_full[:, bold_start:bold_end]
print("Segment:", segment.shape)
np.save(str(OUT_DIR / "segment.npy"), segment)

# SLIDING WINDOW
windows = []
times = []

for t in range(0, segment.shape[1] - WIN + 1, STEP):
    win = segment[:, t:t+WIN]
    windows.append(win)

    win_start_bold = bold_start + t
    win_start_time = win_start_bold * TR
    times.append(win_start_time)

windows = np.array(windows, dtype=np.float32)
times   = np.array(times, dtype=float)

print("Windows:", windows.shape)
print("Time range:", times.min(), "→", times.max())

windows = windows[..., np.newaxis]  # (N, vox, 10, 1)

# PREDICT
preds = model.predict(windows, batch_size=16, verbose=1).reshape(-1)
binary_preds = (preds >= 0.5).astype(int)

np.save(str(OUT_DIR / "predictions.npy"), preds)
np.save(str(OUT_DIR / "predictions_binary.npy"), binary_preds)

# GROUND TRUTH ALIGNMENT
gt_time, gt_binary = create_ground_truth_timeline(
    gt_onsets_unshifted, gt_durations, times.min(), times.max()
)

gt_aligned = np.zeros(len(times), dtype=np.int32)
for i, t in enumerate(times):
    idx = np.argmin(np.abs(gt_time - t))
    gt_aligned[i] = int(gt_binary[idx])

# PERFORMANCE METRICS
precision = precision_score(gt_aligned, binary_preds, zero_division=0)
recall = recall_score(gt_aligned, binary_preds, zero_division=0)
f1 = f1_score(gt_aligned, binary_preds, zero_division=0)

print("\nMETRICS:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# Probability plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(times, preds, label="CNN Probability")
ax.fill_between(gt_time, 0, gt_binary, alpha=0.3, label="GT")
ax.axhline(0.5, linestyle="--", alpha=0.5)
ax.set_title("Predicted Probability vs Ground Truth")
ax.set_xlabel("Time (s, lagged BOLD time)")
ax.set_ylabel("P(face)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(str(OUT_DIR / "plot_probability.png"))
plt.close()

# Binary plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(times, binary_preds, drawstyle="steps-post", linewidth=1.5, label="CNN Binary")
ax.fill_between(gt_time, 0, gt_binary, alpha=0.3, label="GT")
ax.set_title("Binary Predictions vs Ground Truth")
ax.set_xlabel("Time (s, lagged BOLD time)")
ax.set_ylabel("Prediction (0 or 1)")
ax.set_ylim(-0.1, 1.1)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(str(OUT_DIR / "plot_binary.png"))
plt.close()

print("\nAll results saved to:", OUT_DIR)
