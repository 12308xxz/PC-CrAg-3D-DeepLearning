"""Validate split file counts, class labels and cross-split patient ID overlap.

Does NOT inspect patient-level clinical files or claim that filenames certify pathology.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

EXPECTED = {
    "train": {0: 42, 1: 73},
    "internal_val": {0: 18, 1: 31},
    "external_test": {0: 16, 1: 30},
}


def scan(data_root, strict=True):
    all_ids = {}
    errors = []
    for split, classes in EXPECTED.items():
        counts = Counter()
        split_ids = set()
        for label, folder in [(0, "negative"), (1, "positive")]:
            root = Path(data_root) / split / folder
            files = sorted(root.glob("*.npz")) if root.is_dir() else []
            counts[label] = len(files)
            for file in files:
                pid = file.stem
                if pid in split_ids:
                    errors.append(f"Duplicate patient ID within {split}: {pid}")
                split_ids.add(pid)
                # If a file was created with preprocess_ct.py, verify embedded label.
                try:
                    with np.load(file, allow_pickle=False) as item:
                        if "label" in item and int(item["label"]) != label:
                            errors.append(f"Folder/embedded label mismatch: {file}")
                        if "patient_id" in item and str(item["patient_id"].item()) != pid:
                            errors.append(f"Filename/embedded patient_id mismatch: {file}")
                except Exception as exc:
                    errors.append(f"Cannot read {file}: {type(exc).__name__}: {exc}")
        all_ids[split] = split_ids
        print(f"{split}: n={sum(counts.values())}, CrAg-positive={counts[1]}, negative={counts[0]}")
        if counts != Counter(classes):
            errors.append(f"Unexpected counts in {split}: {dict(counts)}; expected {classes}")
    keys = list(all_ids)
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            overlap = all_ids[keys[i]] & all_ids[keys[j]]
            if overlap:
                errors.append(f"Patient ID overlap: {keys[i]} vs {keys[j]} ({len(overlap)} IDs)")
    for error in errors:
        print("WARNING:",error)
    if not errors:
        print("All file-count, class-label, and patient-ID checks passed.")
    if strict and errors:
        raise SystemExit(f"Checks failed ({len(errors)} issue(s)).")
    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    scan(args.data_root, strict=not args.no_strict)
