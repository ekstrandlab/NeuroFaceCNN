#!/usr/bin/env python3
"""
Create Grand Average (Group Level)
This script:
- Scans the 'IG_averaged' folder
- Loads the averaged NIfTI file for every subject.
- Computes the mean across all subjects.
- Saves a single 'Grand Mean' 4D NIfTI file.
"""

import os
import numpy as np
import nibabel as nib

# CONFIG
LABEL = "noface"

# Path Setup
PROJECT_ROOT = r"D:\Sara-Projects\Face-Project\cnn-on-80-participant"
INPUT_DIR = os.path.join(PROJECT_ROOT, "IG", "IG_averaged", LABEL)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "IG", "IG_group_results", LABEL)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# list NIfTI files
def list_niftis(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".nii", ".nii.gz"))
    ]

# Find all subject-average files
subject_files = list_niftis(INPUT_DIR)

if not subject_files:
    raise RuntimeError(f"No subject average files found in {INPUT_DIR}")

print(f"Found {len(subject_files)} subject averages. Calculating Group Mean...\n")


# Load first file to get shape/affine
img0 = nib.load(subject_files[0])
data0 = img0.get_fdata()
X, Y, Z, T = data0.shape

sum_buffer = np.zeros((T, X, Y, Z), dtype=np.float64)
valid_count = 0

# Summation Loop
for fpath in subject_files:
    try:
        img = nib.load(fpath)
        
        # Check Shape
        if img.shape != (X, Y, Z, T):
            print(f"[SKIP] Shape mismatch: {os.path.basename(fpath)} is {img.shape}")
            continue
            
        d = img.get_fdata().astype(np.float64)
        np.nan_to_num(d, copy=False) # Guard against NaNs

        # Add to total (transpose T to front for efficiency)
        sum_buffer += np.moveaxis(d, -1, 0)
        valid_count += 1

    except Exception as e:
        print(f"[ERR] Failed to load {os.path.basename(fpath)}: {e}")


if valid_count == 0:
    raise RuntimeError("No valid files were averaged.")

print(f"\n Computing final mean from {valid_count} subjects...")
group_mean = sum_buffer / valid_count

# Move T back to end: (T, X, Y, Z) -> (X, Y, Z, T)
group_mean = np.moveaxis(group_mean, 0, -1).astype(np.float32)

# Save
out_name = f"Group_Average_{LABEL}_IG.nii.gz"
out_path = os.path.join(OUTPUT_DIR, out_name)

out_img = nib.Nifti1Image(group_mean, affine=img0.affine, header=img0.header)
nib.save(out_img, out_path)

print(f" Group mean saved to: \n {out_path}")

