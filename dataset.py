"""3D CT dataset for image-only and image+clinical models.

Directory layout::

    data_root/
      train/{negative,positive}/*.npz
      internal_val/{negative,positive}/*.npz
      external_test/{negative,positive}/*.npz

Every NPZ must contain keys ``volume`` ([D, H, W]) and ``patient_id``.
For fusion models, provide a CSV containing ``patient_id`` plus exactly the
8 selected, already imputed and training-standardized clinical features.

Training-only augmentation consists of:
three independent flips (p=0.5 each), then optional N(0, 0.01) noise (p=0.3).
No rotations, gamma changes, cutout, or additional transforms are applied.
"""

import csv
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class Volume3DDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        train_mode: bool = False,
        seed: int = 62,
        clinical_csv: Optional[str] = None,
        is_fusion: bool = False,
        clinical_feature_cols: Optional[Sequence[str]] = None,
        patient_id_col: str = "patient_id",
    ):
        self.root = root
        self.split = split
        self.train_mode = train_mode
        self.rng = random.Random(seed)
        # train.py supplies clinical_csv for all architectures, but only fusion
        # uses the features.  The explicit is_fusion flag matches train.py.
        self.is_fusion = bool(is_fusion)
        self.use_clinical = self.is_fusion

        self.class_to_idx = {"negative": 0, "positive": 1}
        self.samples = []
        split_root = os.path.join(root, split)
        if not os.path.exists(split_root):
            raise FileNotFoundError(f"Split not found: {split_root}")

        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(split_root, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, fname)
                if os.path.isfile(path) and fname.lower().endswith(".npz"):
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise ValueError(f"No .npz files found under: {split_root}")

        self.clinical_feature_cols = None
        self.clinical_by_id = {}
        if self.use_clinical:
            if not clinical_csv or not os.path.isfile(clinical_csv):
                raise FileNotFoundError(f"Clinical CSV not found: {clinical_csv}")
            with open(clinical_csv, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or patient_id_col not in reader.fieldnames:
                    raise ValueError(
                        f"Clinical CSV must have a '{patient_id_col}' column."
                    )

                if clinical_feature_cols is None:
                    # Automatic inference works for a minimal clinical CSV.
                    # Pass clinical_feature_cols explicitly if other columns exist.
                    exclude = {patient_id_col, "label", "split"}
                    cols = [c for c in reader.fieldnames if c not in exclude]
                else:
                    cols = list(clinical_feature_cols)
                if len(cols) != 8 or len(set(cols)) != 8:
                    raise ValueError(
                        "The fusion model requires exactly 8 DISTINCT clinical "
                        "feature columns; supply clinical_feature_cols if needed."
                    )
                missing_cols = [c for c in cols if c not in reader.fieldnames]
                if missing_cols:
                    raise ValueError(f"Missing clinical columns: {missing_cols}")
                self.clinical_feature_cols = cols

                for line_no, row in enumerate(reader, start=2):
                    patient_id = str(row[patient_id_col]).strip()
                    if not patient_id:
                        raise ValueError(f"Empty patient_id in CSV line {line_no}")
                    if patient_id in self.clinical_by_id:
                        raise ValueError(
                            f"Duplicate patient_id '{patient_id}' in clinical CSV"
                        )
                    try:
                        values = np.asarray([float(row[c]) for c in cols], dtype=np.float32)
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Non-numeric/missing clinical values for {patient_id} "
                            f"at CSV line {line_no}; impute first."
                        ) from e
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"NaN/Inf clinical values for {patient_id}; impute first."
                        )
                    self.clinical_by_id[patient_id] = values

    def __len__(self):
        return len(self.samples)

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        # vol: [D, H, W]; apply independent spatial flips.
        if self.rng.random() < 0.5:
            vol = vol[:, :, ::-1].copy()  # flip W
        if self.rng.random() < 0.5:
            vol = vol[:, ::-1, :].copy()  # flip H
        if self.rng.random() < 0.5:
            vol = vol[::-1, :, :].copy()  # flip D

        # Add mild Gaussian noise with probability 0.3.
        if self.rng.random() < 0.3:
            noise = np.random.normal(0, 0.01, size=vol.shape).astype(np.float32)
            vol = np.clip(vol + noise, 0.0, 1.0)
        return vol

    def __getitem__(self, index):
        path, label = self.samples[index]
        with np.load(path, allow_pickle=False) as data:
            vol = data["volume"].astype(np.float32)  # [D, H, W]
            pid_value = data["patient_id"]
            patient_id = str(pid_value.item() if pid_value.ndim == 0 else pid_value)

        if self.train_mode:
            vol = self._augment(vol)

        # Normalize intensities before model input.
        vol = (vol - 0.5) / 0.25
        vol = torch.from_numpy(vol).unsqueeze(0).float()  # [1, D, H, W]
        label = torch.tensor(label, dtype=torch.long)

        # Image-only return interface required by train.py.
        if not self.is_fusion:
            return vol, label, patient_id

        if patient_id not in self.clinical_by_id:
            raise KeyError(
                f"No clinical features for patient_id='{patient_id}' in "
                f"CSV (file: {os.path.basename(path)})."
            )
        clinical = torch.from_numpy(self.clinical_by_id[patient_id].copy()).float()
        # Fusion interface: image, 8 clinical features, label, patient ID.
        return vol, clinical, label, patient_id
