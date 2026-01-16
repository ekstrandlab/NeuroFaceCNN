#!/usr/bin/env python3
"""
Create Grand Average (Group Level)

This script:
- Scans the IG-averaged folder
- Loads the averaged NIfTI file for every subject
- Computes the mean across all subjects
- Saves a single group-level 4D NIfTI file
"""

import numpy as np
import nibabel as nib
from pathlib import Path


# CONFIG
LABEL = "face"   # "face" or "noface"

PROJECT_ROOT = Path(__file__).resolve().parent

IG_ROOT = PROJECT_ROOT / "Output" / "IG-result"
INPUT_DIR = IG_ROOT / "IG-averaged" / LABEL
OUTPUT_DIR = IG_ROOT / "IG-group-results" / LABEL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Collect subject-average NIfTIs
if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

subject_files = sorted(
    list(INPUT_DIR.glob("*.nii")) + list(INPUT_DIR.glob("*.nii.gz"))
)

if not subject_files:
    raise RuntimeError(f"No subject average files found in {INPUT_DIR}")

print(f"Found {len(subject_files)} subject averages. Calculating group mean...\n")


# Initialize accumulator
img0 = nib.load(str(subject_files[0]))
data0 = img0.get_fdata()

if data0.ndim != 4:
    raise RuntimeError("Subject-average file is not 4D.")

X, Y, Z, T = data0.shape
sum_buffer = np.zeros((T, X, Y, Z), dtype=np.float64)
valid_count = 0


# Summation loop
for fp in subject_files:
    try:
        img = nib.load(str(fp))

        if img.shape != (X, Y, Z, T):
            print(f"[SKIP] Shape mismatch: {fp.name} {img.shape}")
            continue

        d = img.get_fdata().astype(np.float64)
        np.nan_to_num(d, copy=False)

        sum_buffer += np.moveaxis(d, -1, 0)  # (T, X, Y, Z)
        valid_count += 1

    except Exception as e:
        print(f"[ERR] Failed to load {fp.name}: {e}")

if valid_count == 0:
    raise RuntimeError("No valid files were averaged.")

print(f"\nComputing final group mean from {valid_count} subjects...")


# Finalize mean
group_mean = sum_buffer / valid_count
group_mean = np.moveaxis(group_mean, 0, -1).astype(np.float32)  # (X,Y,Z,T)


# Save
out_name = f"Group_Average_{LABEL}_IG.nii.gz"
out_path = OUTPUT_DIR / out_name

out_img = nib.Nifti1Image(group_mean, affine=img0.affine, header=img0.header)
nib.save(out_img, str(out_path))

print(f"\nGroup mean saved to:\n{out_path}")
