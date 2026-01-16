#!/usr/bin/env python3

import os
import shutil
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nilearn import plotting, surface, datasets, image


PROJECT_ROOT = Path(__file__).resolve().parent

IG_ROOT = PROJECT_ROOT / "Output" / "IG-result"

nii_path = IG_ROOT / "ffa_association-test_z_FDR_0.01.nii"

FIG_DIR = IG_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

temp_dir = IG_ROOT / "temp_surf_images"
temp_dir.mkdir(parents=True, exist_ok=True)

out_fig = FIG_DIR / "ffa-raw.png"


# LOAD DATA
if not nii_path.exists():
    raise FileNotFoundError(f"FFA mask not found: {nii_path}")

img = nib.load(str(nii_path))
data = img.get_fdata()
is_4d = (data.ndim == 4)
n_tp = data.shape[-1] if is_4d else 1


# SURFACES
fs = datasets.fetch_surf_fsaverage()

fig, axes = plt.subplots(
    n_tp, 1,
    figsize=(4, 2 * n_tp),
    gridspec_kw={'wspace': 0.05, 'hspace': 0.05}
)
axes = np.atleast_1d(axes)

# MAIN LOOP
for t in range(n_tp):

    vol3d = image.index_img(img, t) if is_4d else img
    crops = []

    for hemi in ("left", "right"):

        tex = surface.vol_to_surf(
            vol3d,
            fs.pial_left if hemi == "left" else fs.pial_right
        )

        # display-only threshold
        nz = tex[np.nonzero(tex)]
        thr = np.percentile(np.abs(nz), 84) if nz.size else 0.0

        tmp_png = temp_dir / f"h-{hemi}_t{t}.png"

        plotting.plot_surf_stat_map(
            fs.infl_left if hemi == "left" else fs.infl_right,
            tex,
            hemi=hemi,
            view="ventral",
            threshold=thr,
            cmap="Reds",
            bg_map=fs.sulc_left if hemi == "left" else fs.sulc_right,
            colorbar=False,
            figure=plt.figure(dpi=150)
        )
        plt.savefig(tmp_png, bbox_inches="tight")
        plt.close()

        # rotate + tight crop
        img_rgb = plt.imread(tmp_png)[..., :3]
        img_rgb = np.rot90(img_rgb, 3)
        img_rgb = np.fliplr(img_rgb)

        gray = (img_rgb * 255).astype(np.uint8)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        mask = (gray < 250).astype(np.uint8)

        if np.any(mask):
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            crop = img_rgb[y0:y1, x0:x1]
        else:
            crop = np.zeros((10, 10, 3), dtype=img_rgb.dtype)

        crops.append(crop)

    # side-by-side L | R
    h = min(c.shape[0] for c in crops)
    w = min(c.shape[1] for c in crops)
    combined = np.concatenate(
        [cv2.resize(crops[0], (w, h)),
         cv2.resize(crops[1], (w, h))],
        axis=1
    )

    ax = axes[t]
    ax.imshow(combined)
    ax.axis("off")

    if is_4d:
        ax.text(
            -0.1, 0.5, f"T={t+1}",
            va="center", ha="right",
            fontsize=10, fontweight="bold",
            transform=ax.transAxes
        )

cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
plt.colorbar(plt.cm.ScalarMappable(cmap="Reds"), cax=cax)

fig.savefig(out_fig, dpi=500, bbox_inches="tight")
plt.close(fig)

# remove temp images
for p in temp_dir.glob("*.png"):
    p.unlink()
temp_dir.rmdir()

print(f"Saved FFA surface figure to: {out_fig}")
