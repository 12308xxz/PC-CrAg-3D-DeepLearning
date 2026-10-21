"""Preprocess paired CT/lesion-mask NRRDs into lesion-context CT crops.

This implementation follows the user-confirmed research preprocessing:
- Resample CT and mask separately to 1 mm isotropic resolution.
- Determine the 3D ROI bounding box from the (nearest-neighbor) mask.
- Enlarge EACH bounding-box dimension by 1.8 (no minimum crop size).
- Clip a boundary-adjusted CT crop, including surrounding tissue.
- Window to [-1000, 400] HU, normalize to [0, 1], and resize to
  (D, H, W) = (32, 128, 128) with linear interpolation.

The mask guides cropping only. Neither a binary-mask channel nor physical
lesion diameter/volume is supplied as an input to the image model.

This script assumes the input CT and mask share physical geometry and uses
shape agreement after resampling as in the original research script; users
must validate their spatial alignment before using the preprocessing output.
Do NOT commit any patient data, names, output NPZ files, or failure logs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def resample_to_isotropic(image, target_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


def window_ct(vol, w_min=-1000, w_max=400):
    vol = np.clip(vol, w_min, w_max)
    return ((vol - w_min) / (w_max - w_min + 1e-8)).astype(np.float32)


def find_3d_bbox(mask):
    zs, ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        return None
    return int(zs.min()), int(zs.max() + 1), int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)


def expand_bbox_3d(bbox, vol_shape, context_scale=1.8):
    """Expand ROI bounding box by 1.8x per axis; no min_crop_size.

    The boundary shifts reproduce the original preprocessing implementation.
    Crop coordinates are NumPy half-open intervals in (D,H,W) order.
    """
    D, H, W = vol_shape
    z1, z2, y1, y2, x1, x2 = bbox
    dz, dy, dx = z2 - z1, y2 - y1, x2 - x1
    cz, cy, cx = (z1 + z2)/2.0, (y1 + y2)/2.0, (x1 + x2)/2.0

    new_d = max(int(round(dz * context_scale)), 1)
    new_h = max(int(round(dy * context_scale)), 1)
    new_w = max(int(round(dx * context_scale)), 1)

    nz1 = int(round(cz - new_d / 2)); nz2 = nz1 + new_d
    ny1 = int(round(cy - new_h / 2)); ny2 = ny1 + new_h
    nx1 = int(round(cx - new_w / 2)); nx2 = nx1 + new_w

    if nz1 < 0:
        nz2 -= nz1
        nz1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0

    if nz2 > D:
        shift = nz2 - D
        nz1 -= shift
        nz2 = D
    if ny2 > H:
        shift = ny2 - H
        ny1 -= shift
        ny2 = H
    if nx2 > W:
        shift = nx2 - W
        nx1 -= shift
        nx2 = W

    nz1, ny1, nx1 = max(0, nz1), max(0, ny1), max(0, nx1)
    nz2, ny2, nx2 = min(D, nz2), min(H, ny2), min(W, nx2)
    return nz1, nz2, ny1, ny2, nx1, nx2


def resize_3d_volume(vol, out_size=(32, 128, 128)):
    in_d, in_h, in_w = vol.shape
    out_d, out_h, out_w = out_size
    img = sitk.GetImageFromArray(vol.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetSize((out_w, out_h, out_d))
    resampler.SetOutputSpacing((in_w / out_w, in_h / out_h, in_d / out_d))
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    return sitk.GetArrayFromImage(resampler.Execute(img)).astype(np.float32)


def preprocess_one(patient_id, label, split, args, failures):
    class_folder = args.neg_folder if label == 0 else args.pos_folder
    patient_dir = Path(args.raw_root) / class_folder / patient_id
    ct_path, mask_path = patient_dir / args.ct_filename, patient_dir / args.mask_filename
    if not patient_dir.exists():
        failures.append(f"[NoPatientDir] {patient_id} | {patient_dir}")
        return
    if not ct_path.exists():
        failures.append(f"[NoCT] {patient_id} | {ct_path}")
        return
    if not mask_path.exists():
        failures.append(f"[NoMask] {patient_id} | {mask_path}")
        return
    try:
        ct_img = sitk.ReadImage(str(ct_path))
        mask_img = sitk.ReadImage(str(mask_path))
        ct_img = resample_to_isotropic(ct_img, (1.0, 1.0, 1.0), False)
        mask_img = resample_to_isotropic(mask_img, (1.0, 1.0, 1.0), True)
        ct_vol = sitk.GetArrayFromImage(ct_img)
        mask_vol = sitk.GetArrayFromImage(mask_img)
        if ct_vol.shape != mask_vol.shape:
            failures.append(f"[ShapeMismatch] {patient_id} | ct={ct_vol.shape} mask={mask_vol.shape}")
            return
        bbox = find_3d_bbox(mask_vol)
        if bbox is None:
            failures.append(f"[EmptyMask] {patient_id}")
            return
        crop_box = expand_bbox_3d(bbox, ct_vol.shape, context_scale=1.8)
        z1, z2, y1, y2, x1, x2 = crop_box
        crop = ct_vol[z1:z2, y1:y2, x1:x2]
        crop = window_ct(crop, -1000, 400)
        crop = resize_3d_volume(crop, (32, 128, 128))
        class_name = "negative" if label == 0 else "positive"
        out_dir = Path(args.output_root) / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"{patient_id}.npz", volume=crop.astype(np.float32),
            label=np.int64(label), patient_id=patient_id,
            crop_box=np.array(crop_box, dtype=np.int32),
            original_shape=np.array(ct_vol.shape, dtype=np.int32),
        )
    except Exception as exc:
        failures.append(f"[Exception] {patient_id} | {repr(exc)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ROI-guided 3D CT crops without a minimum crop size")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--neg-folder", default="3A")
    parser.add_argument("--pos-folder", default="4A")
    parser.add_argument("--ct-filename", default="lung.nrrd")
    parser.add_argument("--mask-filename", default="lung-label.nrrd")
    args = parser.parse_args()
    df = pd.read_csv(args.split_csv, dtype={"patient_id": str})
    required = {"patient_id", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing CSV columns: {required - set(df.columns)}")
    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs in split CSV")
    if not set(df["label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0/1")
    failures = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Preprocessing"):
        preprocess_one(str(row.patient_id), int(row.label), str(row.split), args, failures)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "build_3d_fail_log.txt").write_text("\n".join(failures), encoding="utf-8")
    print(f"Processed {len(df)-len(failures)}/{len(df)} files. Failures: {len(failures)}")

if __name__ == "__main__":
    main()
