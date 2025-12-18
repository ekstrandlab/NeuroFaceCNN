#!/usr/bin/env python3
"""
Reconstruct voxel-level Integrated Gradients (IG) maps into 4D NIfTI volumes
This script:
- Loads the group-level binary brain mask.
- Loads subject-specific IG attributions stored as numpy arrays.
- Reconstructs each sample (shape: [41489, 10]) back to full 3D brain (85x85x85) over time.
- Saves each reconstructed sample as a 4D NIfTI file
"""

import os
import numpy as np
import nibabel as nib


# CONFIG
LABEL = "face"  # or "noface"

BASE_DIR = r"D:\sarafiles\Face-Project"
MASK_PATH = os.path.join(BASE_DIR, "85_subBrainMask_average_99.nii.gz")

IG_INPUT_DIR = os.path.join(BASE_DIR, "IG_numpy", LABEL)
OUTPUT_DIR = os.path.join(BASE_DIR, "IG_back_to_brain", LABEL)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPECTED_T = 10  # number of TRs per IG sample


# Load brain mask
if not os.path.exists(MASK_PATH):
    raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

mask_img = nib.load(MASK_PATH)
mask_data = mask_img.get_fdata()
mask_shape = mask_data.shape

print(f"Brain mask loaded with shape: {mask_shape}")

mask_flat = mask_data.reshape(-1)
mask_indices = np.nonzero(mask_flat)[0]
print(f"Nonzero (active) voxels in brain mask: {len(mask_indices)}")


# Function to reconstruct full brain volume from masked IG data
def revert_to_brain(voxel_time: np.ndarray) -> nib.Nifti1Image:

    if voxel_time.ndim != 2:
        raise ValueError(f"Expected 2D array, got {voxel_time.ndim}D")

    n_vox, n_time = voxel_time.shape

    if n_vox != len(mask_indices):
        raise ValueError(
            f"Voxel mismatch: data={n_vox}, mask={len(mask_indices)}"
        )

    brain_4d = np.zeros((*mask_shape, n_time), dtype=np.float32)                # (85, 85, 85, 10)

    for t in range(n_time):
        flat_vol = np.zeros(mask_flat.shape, dtype=np.float32)
        flat_vol[mask_indices] = voxel_time[:, t]
        brain_4d[..., t] = flat_vol.reshape(mask_shape)

    return nib.Nifti1Image(brain_4d, affine=mask_img.affine)



# Loop through subjects
sub_folders = sorted(
    d for d in os.listdir(IG_INPUT_DIR)
    if os.path.isdir(os.path.join(IG_INPUT_DIR, d))
)

if not sub_folders:
    raise RuntimeError(f"No subject folders found in {IG_INPUT_DIR}")

print(f"Found {len(sub_folders)} subjects \n")

for subj in sub_folders:
    subj_in = os.path.join(IG_INPUT_DIR, subj)
    subj_out = os.path.join(OUTPUT_DIR, subj)
    os.makedirs(subj_out, exist_ok=True)

    npy_files = sorted(f for f in os.listdir(subj_in) if f.endswith(".npy"))

    if not npy_files:
        print(f"[WARN] No .npy files for {subj}, skipping")
        continue

    print(f"Processing {subj} ({len(npy_files)} samples)")

    for fname in npy_files:
        fpath = os.path.join(subj_in, fname)
        data = np.load(fpath).squeeze()

        if data.shape[1] != EXPECTED_T:
            print(f"[SKIP] {fname}: unexpected T={data.shape[1]}")
            continue

        nifti_img = revert_to_brain(data)

        out_name = fname.replace(".npy", "_IG_backToBrain.nii.gz")
        out_path = os.path.join(subj_out, out_name)
        nib.save(nifti_img, out_path)

    print(f"Finished {subj}\n")

print("All IG reconstructions completed.")
