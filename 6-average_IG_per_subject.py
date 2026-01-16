#!/usr/bin/env python3
"""
Average Reconstructed IG Maps

This script:
- Scans the output folder from IG-backed-to-brain.
- For each subject, loads all reconstructed 4D NIfTI files.
- Computes the element-wise average across all samples.
- Saves a single 4D NIfTI file per subject representing their mean attribution.
"""

import numpy as np
import nibabel as nib
from pathlib import Path

LABEL = "face"  # "face" or "noface"
EXPECTED_T = 10

PROJECT_ROOT = Path(__file__).resolve().parent

IG_ROOT = PROJECT_ROOT / "Output" / "IG-result"
INPUT_DIR = IG_ROOT / "IG-backed-to-brain" / LABEL
OUTPUT_DIR = IG_ROOT / "IG-averaged" / LABEL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# list NIfTI files
def list_niftis(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in [".nii", ".gz"] or p.name.lower().endswith(".nii.gz")])

# Find all subject folders in the input directory
if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

sub_dirs = sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)
if not sub_dirs:
    raise RuntimeError(f"No subject folders found in {INPUT_DIR}")

print(f"Found {len(sub_dirs)} subjects to average.\n")

# Process each subject
for sub_dir in sub_dirs:
    sub = sub_dir.name
    nii_files = sorted(list(sub_dir.glob("*.nii")) + list(sub_dir.glob("*.nii.gz")))

    if not nii_files:
        print(f"[WARN] No NIfTI files for {sub}, skipping.")
        continue

    # Load first file to get dimensions and affine
    img0 = nib.load(str(nii_files[0]))
    data0 = img0.get_fdata()

    # Validation
    if data0.ndim != 4:
        print(f"[ERR] First file for {sub} is not 4D. Skipping.")
        continue

    X, Y, Z, T = data0.shape
    if T != EXPECTED_T:
        print(f"[WARN] {sub} has T={T} (expected {EXPECTED_T}). Processing anyway.")

    # float64 accumulator
    sum_buffer = np.zeros((T, X, Y, Z), dtype=np.float64)
    valid_file_count = 0

    print(f"Processing {sub}: Averaging {len(nii_files)} files...")

    for fp in nii_files:
        try:
            img = nib.load(str(fp))

            if img.shape != (X, Y, Z, T):
                print(f"  [SKIP] Shape mismatch: {fp.name} {img.shape}")
                continue

            d = img.get_fdata().astype(np.float64)
            np.nan_to_num(d, copy=False)

            sum_buffer += np.moveaxis(d, -1, 0)  # (T, X, Y, Z)
            valid_file_count += 1

        except Exception as e:
            print(f"  [ERR] Could not read {fp.name}: {e}")

    if valid_file_count == 0:
        print(f"[FAIL] No valid files averaged for {sub}.")
        continue

    mean_data = (sum_buffer / valid_file_count)
    mean_data = np.moveaxis(mean_data, 0, -1).astype(np.float32)  # back to (X,Y,Z,T)

    out_name = f"{sub}_{LABEL}_avg_IG.nii.gz"
    out_path = OUTPUT_DIR / out_name

    out_img = nib.Nifti1Image(mean_data, affine=img0.affine, header=img0.header)
    nib.save(out_img, str(out_path))

    print(f"[OK] Saved {out_name}\n")

print("All subjects averaged successfully.")
