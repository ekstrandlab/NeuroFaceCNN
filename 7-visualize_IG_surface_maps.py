import os
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, surface, datasets, image
from scipy.ndimage import binary_erosion


BASE_OUT_DIR = Path("/Users/saraasadi/IG_group_results")
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

data_paths = {
    "Face": BASE_OUT_DIR / "face" / "Group_Average_face_IG.nii.gz",
    "NoFace": BASE_OUT_DIR / "noface" / "Group_Average_noface_IG.nii.gz",}

ffa_mask_path = "/Users/saraasadi/ffa_association-test_z_FDR_0.01.nii"

ffa_threshold = 0.5
erosion_iter = 2  # Increase -> make the FFA region smaller 
display_percentile = 0 # No threshold
cmap = "RdBu"

out_png = "group_averaged-IG-0-threshold-ventral.png"

# temp images also live inside output dir
temp_dir = BASE_OUT_DIR / "temp_surf_images"
temp_dir.mkdir(exist_ok=True)

# HELPERS
def load_imgs(paths_dict):
    return {name: nib.load(str(p)) for name, p in paths_dict.items()}

def make_eroded_mask(mask_file, iterations):
    mimg = nib.load(mask_file)
    mask = mimg.get_fdata() > 0
    erode = binary_erosion(mask, iterations=iterations)
    return nib.Nifti1Image(erode.astype(float), mimg.affine)

def proj_mask_to_surf(mask_img, fs, hemi):
    mesh = fs.pial_left if hemi == "left" else fs.pial_right
    proj = surface.vol_to_surf(mask_img, mesh, interpolation="nearest")
    return (proj > ffa_threshold).astype(float)

def proj_vol_to_surf_t(nii4d, t, fs, hemi):
    mesh = fs.pial_left if hemi == "left" else fs.pial_right
    return surface.vol_to_surf(image.index_img(nii4d, t), mesh)

def crop_whitespace(rgb):
    img = np.fliplr(np.rot90(rgb, 3))    # for lateral delete fliplr and rot90 4 
    gray = img.mean(axis=2)
    keep = gray < 0.98
    if not np.any(keep):
        return img
    ys, xs = np.where(keep)
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1, :]

def surf_figure(fs, hemi, tex, thresh, title=None):
    infl = fs.infl_left if hemi == "left" else fs.infl_right
    sulc = fs.sulc_left if hemi == "left" else fs.sulc_right
    fig = plotting.plot_surf_stat_map(
        infl, tex, hemi=hemi, view="ventral",
        threshold=thresh, cmap=cmap, bg_map=sulc,
        colorbar=False, figure=plt.figure(dpi=150)
    )
    if title:
        fig.axes[0].set_title(title, fontsize=10)
    return fig

def add_mask_contour(fs, hemi, mask_vertices, fig):
    if np.count_nonzero(mask_vertices) == 0:
        return
    infl = fs.infl_left if hemi == "left" else fs.infl_right
    try:
        plotting.plot_surf_contours(
            infl, mask_vertices,
            levels=[1], colors=["lime"],
            linewidths=0.01, figure=fig
        )
    except ValueError:
        plotting.plot_surf_stat_map(
            infl, mask_vertices, hemi=hemi, view="ventral",
            threshold=0.5, cmap="Greens", alpha=0.3,
            colorbar=False, figure=fig
        )

def safe_percentile(arr, p):
    arr = np.asarray(arr)
    arr = arr[np.nonzero(arr)]
    return np.percentile(np.abs(arr), p) if arr.size else 0.0

# MAIN
if __name__ == "__main__":

    fs = datasets.fetch_surf_fsaverage()
    imgs = load_imgs(data_paths)

    n_tp = imgs["Face"].shape[-1]

    ffa_mask_img = make_eroded_mask(ffa_mask_path, erosion_iter)
    ffa_surf = {
        hemi: proj_mask_to_surf(ffa_mask_img, fs, hemi)
        for hemi in ("left", "right")
    }

    for cond, nii in imgs.items():

        ncols = int(np.ceil(np.sqrt(n_tp)))
        nrows = int(np.ceil(n_tp / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3 * ncols, 3 * nrows),
            gridspec_kw={"wspace": 0.05, "hspace": 0.15}
        )

        if n_tp == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f"{cond} onset", fontsize=16, fontweight="bold", y=0.98)

        for t in range(n_tp):

            row, col = divmod(t, ncols)
            crops = []

            for hemi in ("left", "right"):
                tex = proj_vol_to_surf_t(nii, t, fs, hemi)
                thresh = safe_percentile(tex, display_percentile)

                f = surf_figure(fs, hemi, tex, thresh)
                add_mask_contour(fs, hemi, ffa_surf[hemi], f)

                tmp = temp_dir / f"{cond}_{hemi}_t{t}.png"
                plt.savefig(tmp, bbox_inches="tight")
                plt.close()

                rgb = plt.imread(tmp)[..., :3]
                crops.append(crop_whitespace(rgb))

            h = min(c.shape[0] for c in crops)
            w = min(c.shape[1] for c in crops)
            crops = [c[:h, :w] for c in crops]

            panel = np.concatenate(crops, axis=1)

            ax = axes[row, col]
            ax.imshow(panel)
            ax.axis("off")

            ax.text(0.5, 1.05, f"T={t+1}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                    transform=ax.transAxes)

            ax.text(0.25, -0.05, "L",
                    ha="center", va="top",
                    fontsize=9, fontweight="bold",
                    transform=ax.transAxes)

            ax.text(0.75, -0.05, "R",
                    ha="center", va="top",
                    fontsize=9, fontweight="bold",
                    transform=ax.transAxes)

        for t in range(n_tp, nrows * ncols):
            axes[t // ncols, t % ncols].axis("off")

        out_file = BASE_OUT_DIR / out_png.replace(".png", f"_{cond}.png")
        fig.savefig(out_file, dpi=500, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_file}")

    # cleanup
    for p in temp_dir.glob("*.png"):
        p.unlink()
    temp_dir.rmdir()
    
