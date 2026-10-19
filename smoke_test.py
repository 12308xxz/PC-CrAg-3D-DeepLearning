"""Synthetic checks for dataset/train interfaces and three model forward passes.

No private patient data, no model weights, and no pretrained weight downloads.
"""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch

from dataset import Volume3DDataset
from models import create_model_3d_baseline, create_model_3d_optimized, create_model_3d_clinical
from train import build_loaders, run_epoch, FocalLoss


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        data_root = base / "processed"
        records = []
        for split in ("train", "internal_val", "external_test"):
            for class_name, label in (("negative", 0), ("positive", 1)):
                pid = f"SYNTH_{split}_{class_name}"
                folder = data_root / split / class_name
                folder.mkdir(parents=True)
                np.savez_compressed(
                    folder / f"{pid}.npz",
                    volume=np.full((16, 64, 64), 0.25 if label == 0 else 0.75, dtype=np.float32),
                    patient_id=pid, label=np.int64(label),
                )
                records.append({"patient_id": pid, **{f"feature_{i}": i * 0.1 for i in range(1, 9)}})
        clinical_csv = base / "clinical.csv"
        pd.DataFrame(records).to_csv(clinical_csv, index=False)
        config = {
            "data_root": str(data_root), "clinical_csv": str(clinical_csv),
            "batch_size": 2, "seed": 62,
        }
        # Verify the specific construction calls used by train.py in BOTH modes.
        for is_fusion in (False, True):
            train_ds, val_ds, train_loader, val_loader = build_loaders(config, is_fusion)
            assert len(train_ds) == len(val_ds) == 2
            batch = next(iter(val_loader))
            assert len(batch) == (4 if is_fusion else 3)
            assert batch[0].shape == (2, 1, 16, 64, 64)
            if is_fusion:
                assert batch[1].shape == (2, 8)
            print("PASS train.build_loaders:", "fusion" if is_fusion else "image-only")

        # Confirm image-only training does not require the clinical CSV file.
        missing_csv = str(base / "not_provided.csv")
        ds = Volume3DDataset(str(data_root), "train", clinical_csv=missing_csv,
                             is_fusion=False, train_mode=True, seed=62)
        assert len(ds[0]) == 3
        print("PASS image-only mode without clinical CSV")

        img, label, pid = Volume3DDataset(str(data_root), "train", train_mode=False)[0]
        clin = Volume3DDataset(str(data_root), "train", is_fusion=True,
                               clinical_csv=str(clinical_csv), train_mode=False)[0][1]
        with torch.inference_mode():
            for name, model in (
                ("baseline", create_model_3d_baseline(pretrained=False)),
                ("optimized", create_model_3d_optimized(pretrained=False)),
                ("fusion", create_model_3d_clinical(pretrained=False)),
            ):
                model.eval()
                pred = model(img.unsqueeze(0), clin.unsqueeze(0)) if name == "fusion" else model(img.unsqueeze(0))
                assert pred.shape == (1, 2), (name, pred.shape)
                print("PASS model forward:", name, tuple(pred.shape))

        # Check run_epoch unpacking follows train.py's call contract.
        class MiniImageModel(torch.nn.Module):
            def forward(self, x):
                v = x.mean(dim=(1, 2, 3, 4))
                return torch.stack([-v, v], dim=1)
        class MiniFusionModel(torch.nn.Module):
            def forward(self, x, clinical):
                v = x.mean(dim=(1, 2, 3, 4)) + clinical[:, 0]
                return torch.stack([-v, v], dim=1)
        criterion = FocalLoss(alpha=[0.55, 0.45], gamma=2.0)
        for fusion in (False, True):
            _, _, _, loader = build_loaders(config, fusion)
            model = MiniFusionModel() if fusion else MiniImageModel()
            loss, prob, labels, pids = run_epoch(
                model, loader, criterion, None, "cpu", fusion, train=False, desc="Synthetic check"
            )
            assert len(prob) == len(labels) == len(pids) == 2
            assert np.isfinite(loss) and np.isfinite(prob).all()
            print("PASS train.run_epoch:", "fusion" if fusion else "image-only")
        print("PASS ALL synthetic checks; no published results or weight files verified")


if __name__ == "__main__":
    main()
