"""Resample CT and lesion mask, crop lesion context, and write compressed NPZ.

Requires inputs in NRRD format and split CSV columns patient_id,label,split.
Patient identifiers are read locally and MUST NOT be distributed in public examples.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def resample_to_isotropic(image, target_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing, original_size = image.GetSpacing(), image.GetSize()
    new_size = [max(1, int(round(original_size[i] * original_spacing[i] / target_spacing[i])))
                for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


def window_ct(volume, low=-1000, high=400):
    volume = np.clip(volume, low, high)
    return ((volume - low) / (high - low + 1e-8)).astype(np.float32)


def find_3d_bbox(mask):
    zs, ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(zs.min()), int(zs.max() + 1), int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)


def expand_bbox_3d(bbox, vol_shape, context_scale=1.8, min_crop_size=(16, 96, 96)):
    D, H, W = vol_shape
    z1, z2, y1, y2, x1, x2 = bbox
    dz, dy, dx = z2 - z1, y2 - y1, x2 - x1
    cz, cy, cx = (z1 + z2)/2, (y1 + y2)/2, (x1 + x2)/2
    nd = max(int(round(dz * context_scale)), min_crop_size[0])
    nh = max(int(round(dy * context_scale)), min_crop_size[1])
    nw = max(int(round(dx * context_scale)), min_crop_size[2])
    nz1, ny1, nx1 = (int(round(cz - nd/2)), int(round(cy - nh/2)), int(round(cx - nw/2)))
    nz2, ny2, nx2 = nz1 + nd, ny1 + nh, nx1 + nw
    if nz1 < 0: nz2 -= nz1; nz1 = 0
    if ny1 < 0: ny2 -= ny1; ny1 = 0
    if nx1 < 0: nx2 -= nx1; nx1 = 0
    if nz2 > D: nz1 -= nz2 - D; nz2 = D
    if ny2 > H: ny1 -= ny2 - H; ny2 = H
    if nx2 > W: nx1 -= nx2 - W; nx2 = W
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
    if not ct_path.exists() or not mask_path.exists():
        failures.append(f"Missing CT or mask: {patient_id}")
        return
    try:
        ct = sitk.ReadImage(str(ct_path))
        mask = sitk.ReadImage(str(mask_path))
        # Assumes original CT and mask share the same physical geometry.
        # Verify alignment before resampling, rather than silently misaligning them.
        if (ct.GetSize() != mask.GetSize()
                or not np.allclose(ct.GetSpacing(), mask.GetSpacing())
                or not np.allclose(ct.GetOrigin(), mask.GetOrigin())
                or not np.allclose(ct.GetDirection(), mask.GetDirection())):
            failures.append(f"Geometry mismatch: {patient_id}")
            return
        ct = resample_to_isotropic(ct, (1.0, 1.0, 1.0), False)
        mask = resample_to_isotropic(mask, (1.0, 1.0, 1.0), True)
        ct_vol, mask_vol = sitk.GetArrayFromImage(ct), sitk.GetArrayFromImage(mask)
        if ct_vol.shape != mask_vol.shape:
            failures.append(f"Resampled shape mismatch: {patient_id}")
            return
        bbox = find_3d_bbox(mask_vol)
        if bbox is None:
            failures.append(f"Empty mask: {patient_id}")
            return
        crop_box = expand_bbox_3d(bbox, ct_vol.shape, context_scale=1.8,
                                  min_crop_size=(16, 96, 96))
        z1, z2, y1, y2, x1, x2 = crop_box
        crop = window_ct(ct_vol[z1:z2, y1:y2, x1:x2])
        crop = resize_3d_volume(crop, (32, 128, 128))
        class_name = "negative" if label == 0 else "positive"
        out_dir = Path(args.output_root) / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{patient_id}.npz",
                            volume=crop, label=np.int64(label), patient_id=str(patient_id),
                            crop_box=np.asarray(crop_box, dtype=np.int32),
                            original_shape=np.asarray(ct_vol.shape, dtype=np.int32))
    except Exception as exc:
        failures.append(f"Failed patient {patient_id}: {type(exc).__name__}: {exc}")


def main():
    p = argparse.ArgumentParser(description="Build 3D ROI crops from NRRD CT/masks")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--output-root", default="data/processed")
    p.add_argument("--neg-folder", default="3A")
    p.add_argument("--pos-folder", default="4A")
    p.add_argument("--ct-filename", default="lung.nrrd")
    p.add_argument("--mask-filename", default="lung-label.nrrd")
    args = p.parse_args()
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
    # IDs in failure logs are private; file is stored locally and ignored by git.
    (Path(args.output_root) / "build_3d_fail_log.txt").write_text("\n".join(failures), encoding="utf-8")
    print(f"Processed {len(df) - len(failures)}/{len(df)} files. Failures: {len(failures)}")


if __name__ == "__main__":
    main()
