# Pulmonary cryptococcosis: dominant-lesion CT and qualitative serum CrAg

**Source-code companion for a retrospective two-center, 3D CT deep learning study.**

This repository provides the **3D image and image–clinical modeling pipeline** used to investigate the association between the dominant-lesion CT phenotype and the observed *qualitative* serum cryptococcal antigen (CrAg) status in patients with pathologically confirmed pulmonary cryptococcosis. The labels represent routine positive/negative serum CrAg results, **not independently adjudicated false-negative or false-positive tests**. This exploratory modeling study does not validate CT as a substitute for direct serum CrAg testing.

> **Scope and limitations:** This code bundle contains *no patient data, CT scans, segmentations, clinical CSVs, patient-level predictions, pretrained weight files, or fine-tuned checkpoints*. It includes models, preprocessing, training, and basic checks. It does **not** contain every analysis needed to regenerate the manuscript's tables and figures (e.g., radiomics/clinical-only comparators, DeLong/McNemar analyses, calibration figures, and Grad-CAM scripts). The private research data and verified final checkpoints are required to verify the paper's published performance; the synthetic smoke test is **not** such a verification. The `dataset.py` module supports both image-only and image–clinical fusion models. It implements training-only augmentation consisting of independent random flips along the three spatial axes and additive Gaussian noise. The dataset interfaces are compatible with the provided `train.py`.

## Repository files

| File | Purpose |
| --- | --- |
| `preprocess_ct.py` | Resample an already annotated CT and 3D ROI mask, then save a normalized lesion-context crop as NPZ. |
| `dataset.py` | Read CT NPZ files; apply the specified augmentations *only during training*; optionally join eight prepared clinical features by patient ID. |
| `models.py` | Baseline 3D ResNet-18, optimized 3D ResNet-18 (small stem and SE), and 3D+clinical fusion model. |
| `train.py` | Training, development-set checkpoint/threshold selection, predictions, and evaluation of an optional independent external split. |
| `check_splits.py` | Check NPZ counts and class/ID consistency across locally supplied splits. |
| `inspect_checkpoint.py` | Check architecture compatibility and saved metadata for a **trusted** author-created checkpoint. |
| `smoke_test.py` | Synthetic, local integration test for image-only and fusion data interfaces and model forward passes. |
| `requirements.txt` | Runtime dependencies (intentionally not presented as an exact environment lockfile). |
| `.gitignore` | Exclude clinical data, images, weights, and generated outputs from Git commits. |

## Environment

The study analysis reported **Python 3.9**. Install a mutually compatible set of PyTorch and torchvision versions, then:

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python train.py --help
```

The dependency list in `requirements.txt` is **not** an exact version-locked environment specification. `torchvision.models.video.r3d_18` and `R3D_18_Weights.KINETICS400_V1` must be supported. Models use Kinetics-400 pretraining *by default during training*, so the first run may download torchvision's generic pretrained weights to the user's local cache. **No pretrained or fine-tuned weight file is included in this archive.** The smoke test explicitly uses `pretrained=False` and does not download weights.

## Inputs and protected data

This repository does not publish the private patient-level files. The preprocessing script accepts paired, spatially aligned, already-segmented NRRD CT images and target-lesion masks. A local split manifest should contain:

```text
patient_id,label,split
SYNTHETIC_NEG,0,train
SYNTHETIC_POS,1,train
```

The example IDs are **fictional**. Valid split names used by `train.py` are `train`, `internal_val` (the manuscript's **development** cohort), and `external_test` (independent external validation). `preprocess_ct.py` defaults to class-coded source directories `3A/` (negative) and `4A/` (positive), each containing a patient-ID subdirectory with `lung.nrrd` and `lung-label.nrrd`; override these names by CLI options if necessary. The script requires matching CT/mask geometry and existing annotations; it does not segment lesions.

```bash
python preprocess_ct.py \
  --raw-root PATH_TO_PRIVATE_CT \
  --split-csv PATH_TO_PRIVATE_SPLITS.csv \
  --output-root data/processed
python check_splits.py --data-root data/processed
```

The expected cohort totals are:

| Local directory | Negative (0) | Positive (1) | Total |
| --- | ---: | ---: | ---: |
| `train` | 42 | 73 | 115 |
| `internal_val` (development) | 18 | 31 | 49 |
| `external_test` | 16 | 30 | 46 |

After preprocessing, input files have this layout:

```text
data/processed/
  train/negative/*.npz
  train/positive/*.npz
  internal_val/negative/*.npz
  internal_val/positive/*.npz
  external_test/negative/*.npz
  external_test/positive/*.npz
```

Each NPZ contains `volume` (a **3D CT image tensor**, not numerical lesion volume), `patient_id`, and auxiliary preprocessing metadata. Patient label comes from its `negative`/`positive` directory. `check_splits.py` checks that the local counts and IDs are consistent; it does **not** certify that those files or labels are the ones used for the published numerical results.

### CT preprocessing

- CT and annotated lesion-mask spacing are resampled to **1 × 1 × 1 mm** (CT linear interpolation, mask nearest-neighbor).
- CT values are clipped to **−1000 through 400 HU** and scaled to `[0, 1]`.
- The annotated dominant-lesion bounding box is expanded by a factor of **1.8**, subject to minimum crop dimensions **16 × 96 × 96** voxels (D × H × W).
- The extracted crop is resized to **32 × 128 × 128** voxels and saved to `.npz`.
- In `dataset.py`, the volume is normalized as `(volume − 0.5) / 0.25` immediately before it is returned.

### Training-only data augmentation

The dataset applies the following operations in this order:

1. Flip width (W) with probability **0.5**.
2. Flip height (H) with probability **0.5**.
3. Flip depth (D) with probability **0.5**.
4. With probability **0.3**, add independent Gaussian noise with **mean 0**, **standard deviation 0.01** in `[0,1]` intensity units and clip the noisy volume to `[0,1]`.

The flips are independent. **No rotation, gamma adjustment, brightness/intensity scaling or shift, or cutout is used.** `internal_val` and `external_test` have no augmentation. The `random.Random(seed)` state drives augmentation choices, while `np.random.normal` draws from NumPy's global random generator; `train.py` seeds Python and NumPy prior to training. Full bitwise reproducibility may still depend on environment and training order.

## Fusion clinical CSV

For `--model fusion`, supply a *private* CSV with `patient_id` plus **eight** numeric feature columns, in the **feature order used for model training**:

1. Monocyte count
2. hs-CRP
3. D-dimer
4. Cough
5. Sputum production
6. Hemoglobin
7. Eosinophil count
8. Procalcitonin (PCT)

These are conceptual variable names; **verify the clinical CSV column names and feature order used for model training** rather than inferring them from this README. Continuous-variable missing-value imputation and standardization and binary-variable mode imputation must be fitted on the **training cohort only**, then reused unchanged for development and external patients. `dataset.py` does not estimate imputation/scaling parameters. Clinical vectors must already be complete, numeric, and finite. The 8 feature columns are read in their CSV column order. `label` and `split` are ignored if present. With the provided `train.py` CLI, use a minimal CSV (patient ID, eight predictors, and optional `label`/`split` columns); the CLI does not expose the optional `clinical_feature_cols` parameter. Do not include outcome as a predictor.

- **Image-only:** `Volume3DDataset(..., is_fusion=False)` returns `(image, label, patient_id)`.
- **Fusion:** `Volume3DDataset(..., is_fusion=True, clinical_csv=...)` returns `(image, clinical_8_vector, label, patient_id)`.

The training script passes `clinical_csv` in *both* modes but uses the CSV **only when `is_fusion=True`**; running image-only training does **not** require a clinical CSV on disk.

## Run the 3D models

```bash
# Baseline 3D ResNet-18
python train.py --model baseline --data-root data/processed --save-dir runs/baseline_seed62

# Optimized 3D ResNet-18
python train.py --model optimized --data-root data/processed --save-dir runs/optimized_seed62

# 3D + clinical fusion (requires completed clinical CSV)
python train.py --model fusion --data-root data/processed \
  --clinical-csv data/clinical_features_8lasso.csv --save-dir runs/fusion_seed62
```

The three variants all output **two logits** for negative/positive classification. The baseline adapts Kinetics-400-pretrained initial convolution weights to one CT channel by averaging across the input-channel dimension. Optimized and fusion models use a newly initialized small stem with SE blocks in `layer3` and `layer4`, retaining compatible pretrained backbone weights.

The training script's defaults are: seed **62**, Adam, batch size **4**, maximum **80** epochs, weight decay **1×10⁻⁴**, focal loss `gamma=2` with class weights `[0.55, 0.45]` for classes `[negative, positive]`, early-stop patience **12**, and a minimum selection-score improvement of **0.001**. The image-only learning rate is **1×10⁻⁴** with ReduceLROnPlateau; fusion uses backbone **1×10⁻⁵**, clinical/fusion head **1×10⁻⁴**, and five-epoch warmup followed by cosine scheduling.

In `internal_val`, each epoch searches thresholds **0.40, 0.45, 0.50, 0.55, 0.60, 0.65**, maximizing balanced accuracy. Checkpoint selection uses **development AUC + 0.5 × development balanced accuracy**. The selected checkpoint's threshold is stored with it and applied unchanged to the optional external cohort. This threshold grid is a *pragmatic candidate range*, not a medically established decision threshold. The development cohort is involved in model selection and is **not** an independent validation cohort.

A training run writes a local checkpoint (`best_model.pth`) and patient-ID-bearing prediction/metric files to `--save-dir` (such as `runs/...`). **Do not upload these outputs to GitHub.** Because no final model weights or private data are bundled, the published patient-level metrics cannot be independently reproduced by simply downloading this repository.

## Optional trusted checkpoint check

Only if you own and trust a final `.pth` file, you can inspect its metadata and model-state compatibility:

```bash
python inspect_checkpoint.py --checkpoint PATH_TO_YOUR_FINAL_MODEL.pth --model fusion
```

This uses PyTorch checkpoint deserialization and must **not** be run on untrusted files. Do not assume an expected threshold (for example, 0.65) without verifying the actual final saved checkpoint.

## Responsible reporting and reproducibility boundaries

- The image-model source, file formats, CLI, model structure and evaluation workflow can be inspected and tested without participant data.
- Exact reported AUCs, confidence intervals, DeLong/McNemar comparisons, radiomics results, calibration plots, Grad-CAM outputs and clinical-variable selection **cannot** be verified using this code alone; some require other analysis scripts and the withheld participant-level data.
- This archive contains no trained weights of any kind. Optional Kinetics-400 weights are retrieved separately from torchvision when you train with pretrained initialization.
- Subject-level CT, masks, features, IDs and predictions remain restricted by institutional permissions. Check all files before making a public repository.
- The article studies associations in confirmed pulmonary cryptococcosis. It does not validate a clinical decision threshold or adjudicate individual CrAg test errors.
