#!/usr/bin/env python3
"""
Average Reconstructed IG Maps
This script:
- Scans the output folder from the previous script (IG_back_to_brain).
- For each subject, loads all reconstructed 4D NIfTI files.
- Computes the element-wise average across all samples.
- Saves a single 4D NIfTI file per subject representing their mean attribution.
"""

import os
import numpy as np
import nibabel as nib

LABEL = "noface" #or noface


PROJECT_ROOT = r"D:\Sara-Projects\Face-Project"
INPUT_DIR = os.path.join(PROJECT_ROOT, "IG", "IG_back_to_brain", LABEL)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "IG", "IG_averaged", LABEL)

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPECTED_T = 10 

# list NIfTI files
def list_niftis(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".nii", ".nii.gz"))
    ]

# Find all subject folders in the input directory
sub_folders = sorted([
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
])

if not sub_folders:
    raise RuntimeError(f"No subject folders found in {INPUT_DIR}")

print(f"Found {len(sub_folders)} subjects to average.\n")

# Process each subject
for sub in sub_folders:
    sub_dir = os.path.join(INPUT_DIR, sub)
    nii_files = list_niftis(sub_dir)

    if not nii_files:
        print(f"[WARN] No NIfTI files for {sub}, skipping.")
        continue

    # Load first file to get dimensions and affine
    img0 = nib.load(nii_files[0])
    data0 = img0.get_fdata()
    
    # Validation
    if data0.ndim != 4:
        print(f"[ERR] First file for {sub} is not 4D. Skipping.")
        continue
    
    X, Y, Z, T = data0.shape
    if T != EXPECTED_T:
        print(f"[WARN] {sub} has T={T} (expected {EXPECTED_T}). Processing anyway.")


    # float64 accumulator to prevent overflow/precision loss
    sum_buffer = np.zeros((T, X, Y, Z), dtype=np.float64)
    valid_file_count = 0

    print(f"Processing {sub}: Averaging {len(nii_files)} files...")

    for fpath in nii_files:
        try:
            img = nib.load(fpath)
            
            # Quick shape check
            if img.shape != (X, Y, Z, T):
                print(f"  [SKIP] Shape mismatch: {os.path.basename(fpath)} {img.shape}")
                continue

            d = img.get_fdata().astype(np.float64)
            np.nan_to_num(d, copy=False) # Safety: replace NaNs with 0
            
            # Optimization: Move Time axis to front (T, X, Y, Z) for faster summation
            # standard NIfTI is (X,Y,Z,T), we prefer (T,X,Y,Z) for numpy broadcasting
            sum_buffer += np.moveaxis(d, -1, 0)
            valid_file_count += 1
            
        except Exception as e:
            print(f"  [ERR] Could not read {os.path.basename(fpath)}: {e}")

    # Finalize Average 
    if valid_file_count == 0:
        print(f"[FAIL] No valid files averaged for {sub}.")
        continue

    # Divide by N to get mean
    mean_data = sum_buffer / valid_file_count
    
    # Move Time axis back to end -> (X, Y, Z, T)
    mean_data = np.moveaxis(mean_data, 0, -1).astype(np.float32)

    # Save Result
    out_name = f"{sub}_{LABEL}_avg_IG.nii.gz"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    out_img = nib.Nifti1Image(mean_data, affine=img0.affine, header=img0.header)
    nib.save(out_img, out_path)
    
    print(f"[OK] Saved {out_name}\n")

print("All subjects averaged successfully.")
