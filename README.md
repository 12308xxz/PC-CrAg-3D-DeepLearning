# Pulmonary cryptococcosis: 3D CT deep learning for qualitative serum CrAg status

Source code accompanying the two-center study *Association between dominant-lesion CT phenotype and qualitative serum cryptococcal antigen status in pulmonary cryptococcosis: a two-center 3D deep learning study*.

## Scope

This repository provides the **core 3D CT deep learning workflow** for classifying the observed qualitative serum cryptococcal antigen (CrAg) status in patients with pathologically confirmed pulmonary cryptococcosis. It includes ROI-guided CT preprocessing, CT dataset loading, training-time augmentation, baseline and optimized 3D ResNet-18 models, an image–clinical fusion model, training and evaluation scripts, and local code checks.

The image branch takes a three-dimensional CT crop centered on a manually annotated dominant lesion. The ROI mask defines the crop location; the crop contains the lesion **and its surrounding CT context**. The binary lesion mask is not used as an additional model-input channel.

## Repository contents

| File | Purpose |
| --- | --- |
| `preprocess_ct.py` | Resample paired CT/ROI NRRDs and build 3D lesion-context NPZ inputs. |
| `dataset.py` | Load 3D inputs, apply training augmentation, and optionally load prepared clinical vectors. |
| `models.py` | Baseline 3D ResNet-18, optimized 3D ResNet-18 with small stem and SE modules, and image–clinical fusion model. |
| `train.py` | Train models; select checkpoints and thresholds using the development cohort; evaluate available splits. |
| `check_splits.py` | Check locally prepared cohort sizes and sample identifiers. |
| `smoke_test.py` | Run model and data-interface checks on synthetic inputs. |
| `test_preprocessing.py` | Check 1.8-fold ROI-guided crop geometry using synthetic inputs. |
| `inspect_checkpoint.py` | Inspect a trusted, locally saved model checkpoint. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Git exclusions for locally generated data and outputs. |

## Installation

Use a compatible Python, PyTorch and torchvision environment. The study used Python 3.9. After installing PyTorch and torchvision for your platform, run:

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python test_preprocessing.py
python train.py --help
```

The image encoder can be initialized from torchvision's `R3D_18_Weights.KINETICS400_V1` (Kinetics-400 pretraining); torchvision may retrieve those pretrained parameters on first use.

## Input data organization

`preprocess_ct.py` reads CT volumes and corresponding manually annotated masks in NRRD format. The expected local organization is:

```text
raw_root/
  3A/                         # observed CrAg-negative
    PATIENT_ID/
      lung.nrrd
      lung-label.nrrd
  4A/                         # observed CrAg-positive
    PATIENT_ID/
      lung.nrrd
      lung-label.nrrd
```

A CSV assigns patients to splits:

```csv
patient_id,label,split
EXAMPLE_NEG,0,train
EXAMPLE_POS,1,train
```

The sample identifiers above are illustrative. The training script uses `train`, `internal_val` (development), and `external_test` (independent external validation).

## CT preprocessing

For each target lesion, `preprocess_ct.py`:

1. Resamples the CT and its ROI mask to **1 × 1 × 1 mm³** spacing, using linear interpolation for CT and nearest-neighbor interpolation for the mask.
2. Locates the tight three-dimensional bounding box around the annotated dominant lesion.
3. Expands the bounding box by **1.8 times along each spatial axis** and adjusts the crop to the CT image boundaries.
4. Extracts the corresponding **CT image block**, including the lesion and surrounding anatomical context.
5. Clips CT intensities to **−1000 to 400 HU**, normalizes to **[0, 1]**, and linearly resizes the crop to **32 × 128 × 128** (`D × H × W`).
6. Saves the processed CT volume and local preprocessing metadata in a compressed `.npz` file.

The crop dimensions scale with each lesion's ROI bounding box before fixed-matrix resizing. **Physical lesion diameter and volume are not given directly to the 3D image encoder.** The input is the resized CT image crop, rather than masked-only lesion voxels.

Build the local dataset:

```bash
python preprocess_ct.py \
  --raw-root PATH_TO_CT_DIRECTORY \
  --split-csv PATH_TO_SPLIT.csv \
  --output-root data/processed

python check_splits.py --data-root data/processed
```

Processed images use this directory layout:

```text
data/processed/
  train/negative/*.npz
  train/positive/*.npz
  internal_val/negative/*.npz
  internal_val/positive/*.npz
  external_test/negative/*.npz
  external_test/positive/*.npz
```

The research cohort comprises 115 training patients (42 negative, 73 positive), 49 development patients (18 negative, 31 positive), and 46 independent external validation patients (16 negative, 30 positive).

## Training-time data augmentation

Only training images undergo augmentation:

- Independent random flips along each of the three spatial axes (probability **0.5 per axis**).
- Additive Gaussian noise (probability **0.3**; mean **0**, standard deviation **0.01** in normalized intensity units).

Before being passed to the network, image intensities are standardized as `(x - 0.5) / 0.25`. Development and external evaluation images are not augmented.

## Models and clinical inputs

- **Baseline:** 3D ResNet-18 with single-channel CT input and Kinetics-400 initialization.
- **Optimized:** 3D ResNet-18 with a small stem and squeeze-and-excitation (SE) modules in the third and fourth residual stages.
- **Fusion:** optimized 3D image-encoder architecture plus a separate clinical feature branch that combines eight selected clinical features with the CT representation.

For fusion, prepare a local CSV containing `patient_id` and exactly eight numeric clinical predictors in the feature order used during model training. Missing-value imputation and continuous-feature standardization should be fitted on training data and applied consistently to development and external samples before loading the fusion CSV.

## Model training and evaluation

```bash
# Baseline
python train.py --model baseline --data-root data/processed --save-dir runs/baseline

# Optimized
python train.py --model optimized --data-root data/processed --save-dir runs/optimized

# Image–clinical fusion
python train.py --model fusion --data-root data/processed \
  --clinical-csv PATH_TO_PREPARED_CLINICAL_CSV \
  --save-dir runs/fusion
```

Training defaults include random seed **62**, Adam optimization, batch size **4**, a maximum of **80 epochs**, and focal loss with `gamma = 2`, negative/positive class weights `[0.55, 0.45]`. The development cohort is used for early stopping, model selection, and threshold selection. Candidate probability thresholds are **0.40–0.65** in steps of **0.05**; the selected threshold is carried unchanged into external evaluation.

To check the image preprocessing and model interfaces without loading study images, run:

```bash
python test_preprocessing.py
python smoke_test.py
```

## Citation

Please cite the associated *BMC Medical Imaging* manuscript when referring to the study's methods or results.
