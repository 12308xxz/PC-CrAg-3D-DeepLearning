import os
import argparse

import math

import random

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from sklearn.metrics import (

    roc_auc_score, confusion_matrix, classification_report,

    accuracy_score, precision_score, recall_score, f1_score,

)

from torch.utils.data import DataLoader

from tqdm import tqdm

# Only load checkpoints created by you; torch.load(... weights_only=False) uses pickle.

from models import (

    create_model_3d_baseline,

    create_model_3d_optimized,

    create_model_3d_clinical,

)

from dataset import Volume3DDataset

# ================== Focal Loss ==================

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0):

        super().__init__()

        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None

        self.gamma = gamma

    def forward(self, inputs, targets):

        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:

            alpha_t = self.alpha.to(inputs.device)[targets]

            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

# ================== 工具函数 ==================

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

def safe_auc(y_true, y_prob):

    y_true = np.asarray(y_true)

    if len(np.unique(y_true)) < 2:

        return float("nan")

    return roc_auc_score(y_true, y_prob)

def compute_metrics(y_true, y_prob, threshold):

    y_true = np.asarray(y_true)

    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    else:

        sens, spec = 0.0, 0.0

    return {

        "auc": safe_auc(y_true, y_prob),

        "acc": accuracy_score(y_true, y_pred),

        "sens": sens,

        "spec": spec,

        "precision": precision_score(y_true, y_pred, zero_division=0),

        "recall": recall_score(y_true, y_pred, zero_division=0),

        "f1": f1_score(y_true, y_pred, zero_division=0),

        "cm": cm,

    }

def compute_selection_score(metrics):

    """AUC + 0.5 * [(sens + spec) / 2]"""

    auc = 0.0 if np.isnan(metrics["auc"]) else metrics["auc"]

    bal = 0.5 * (metrics["sens"] + metrics["spec"])

    return float(auc + 0.5 * bal)

# 默认阈值网格：0.40, 0.45, 0.50, 0.55, 0.60, 0.65

DEFAULT_THRESHOLD_CANDIDATES = tuple(

    float(x) for x in np.arange(0.40, 0.65 + 0.01, 0.05).round(2)

)

def find_best_threshold(y_true, y_prob, candidates=DEFAULT_THRESHOLD_CANDIDATES):

    """

    在固定候选网格上最大化 selection score：

        AUC + 0.5 * [(sens + spec) / 2]

    因为 AUC 与阈值无关，等价于最大化 balanced accuracy。

    候选网格默认包含 0.65，allows selection of 0.65 when supported by validation predictions.

    """

    best_thr = None

    best_score = -np.inf

    best_metrics = None

    for thr in candidates:

        thr = float(thr)

        m = compute_metrics(y_true, y_prob, thr)

        score = compute_selection_score(m)

        bal = 0.5 * (m["sens"] + m["spec"])

        best_bal = -np.inf if best_metrics is None else 0.5 * (best_metrics["sens"] + best_metrics["spec"])

        if (score > best_score) or (np.isclose(score, best_score) and bal > best_bal):

            best_score = score

            best_thr = thr

            best_metrics = m

    return best_thr, best_metrics, best_score

# ================== 数据加载 ==================

def build_loaders(config, is_fusion):

    train_ds = Volume3DDataset(

        config["data_root"], "train",

        clinical_csv=config["clinical_csv"],

        is_fusion=is_fusion,

        train_mode=True,

        seed=config["seed"],

    )

    val_ds = Volume3DDataset(

        config["data_root"], "internal_val",

        clinical_csv=config["clinical_csv"],

        is_fusion=is_fusion,

        train_mode=False,

        seed=config["seed"],

    )

    train_loader = DataLoader(

        train_ds, batch_size=config["batch_size"], shuffle=True,

        num_workers=0, pin_memory=True,

    )

    val_loader = DataLoader(

        val_ds, batch_size=config["batch_size"], shuffle=False,

        num_workers=0, pin_memory=True,

    )

    return train_ds, val_ds, train_loader, val_loader

# ================== 训练/验证/评估循环 ==================

def run_epoch(model, loader, criterion, optimizer, device, is_fusion, train=True, desc="Train"):

    model.train() if train else model.eval()

    losses = []

    probs, labels, pids = [], [], []

    with torch.set_grad_enabled(train):

        for batch in tqdm(loader, desc=desc, leave=False):

            if is_fusion:

                vols, clinicals, targets, batch_pids = batch

                clinicals = clinicals.to(device)

            else:

                vols, targets, batch_pids = batch

            vols = vols.to(device)

            targets = targets.to(device)

            if train:

                optimizer.zero_grad()

            logits = model(vols, clinicals) if is_fusion else model(vols)

            loss = criterion(logits, targets)

            if train:

                loss.backward()

                optimizer.step()

            losses.append(loss.item())

            probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

            labels.extend(targets.detach().cpu().numpy())

            pids.extend(list(batch_pids))

    return np.mean(losses), np.array(probs), np.array(labels), pids

def evaluate_model(model, data_root, split, config, is_fusion, criterion, device,

                   threshold, desc="Eval"):

    ds = Volume3DDataset(

        data_root, split,

        clinical_csv=config["clinical_csv"],

        is_fusion=is_fusion,

        train_mode=False,

        seed=config["seed"],

    )

    loader = DataLoader(

        ds, batch_size=config["batch_size"], shuffle=False,

        num_workers=0, pin_memory=True,

    )

    _, probs, labels, pids = run_epoch(

        model, loader, criterion, None, device, is_fusion, train=False, desc=desc

    )

    metrics = compute_metrics(labels, probs, threshold)

    return metrics, {"patient_ids": pids, "labels": labels, "probs": probs}, len(ds)

def save_predictions_csv(patient_ids, labels, probs, threshold, out_path):

    preds = (np.asarray(probs) >= threshold).astype(int)

    df = pd.DataFrame({

        "patient_id": patient_ids,

        "label": labels,

        "prob": probs,

        "pred": preds,

    }).sort_values("patient_id").reset_index(drop=True)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

# ================== 主训练流程 ==================

def train_experiment(config):

    device = config["device"]

    is_fusion = config["is_fusion"]

    set_seed(config["seed"])

    os.makedirs(config["save_dir"], exist_ok=True)

    train_ds, val_ds, train_loader, val_loader = build_loaders(config, is_fusion)

    # 模型

    if is_fusion:

        model = create_model_3d_clinical(clinical_dim=config.get("clinical_dim", 8)).to(device)

    elif config["use_optimized"]:

        model = create_model_3d_optimized().to(device)

    else:

        model = create_model_3d_baseline().to(device)

    criterion = FocalLoss(alpha=[0.55, 0.45], gamma=2.0)

    # 优化器 / 差分学习率

    if is_fusion:

        backbone_params = [p for n, p in model.named_parameters()

                           if n.startswith("image_encoder") and p.requires_grad]

        head_params = [p for n, p in model.named_parameters()

                       if not n.startswith("image_encoder") and p.requires_grad]

        optimizer = optim.Adam(

            [

                {"params": backbone_params, "lr": config.get("backbone_lr", 1e-5), "name": "backbone"},

                {"params": head_params, "lr": config.get("head_lr", 1e-4), "name": "head"},

            ],

            weight_decay=config["weight_decay"],

        )

    else:

        optimizer = optim.Adam(

            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],

        )

    # 调度器：融合模型 warm-up + cosine；其它使用 ReduceLROnPlateau

    if is_fusion:

        warmup_epochs = config.get("warmup_epochs", 5)

        total_epochs = config["num_epochs"]

        min_lr_ratio = config.get("min_lr_ratio", 0.01)

        def lr_lambda(epoch):

            if epoch < warmup_epochs:

                return float(epoch + 1) / float(warmup_epochs)

            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)

            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        scheduler_type = "epoch"

    else:

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(

            optimizer, mode="max", factor=0.5, patience=4,

        )

        scheduler_type = "plateau"

    best_score = -np.inf

    patience_counter = 0

    best_path = os.path.join(config["save_dir"], "best_model.pth")

    threshold_candidates = config.get("threshold_candidates", DEFAULT_THRESHOLD_CANDIDATES)

    print("=" * 70)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Fusion: {is_fusion}")

    print(f"Scheduler: {scheduler_type}")

    print(f"Threshold candidates: {threshold_candidates}")

    print("=" * 70)

    for epoch in range(config["num_epochs"]):

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        tr_loss, _, _, _ = run_epoch(

            model, train_loader, criterion, optimizer, device, is_fusion,

            train=True, desc="Train",

        )

        va_loss, va_probs, va_labels, _ = run_epoch(

            model, val_loader, criterion, None, device, is_fusion,

            train=False, desc="Val",

        )

        # 在候选网格上找 best threshold（含 0.65）

        thr, metrics, score = find_best_threshold(va_labels, va_probs, threshold_candidates)

        if scheduler_type == "plateau":

            scheduler.step(0.0 if np.isnan(metrics["auc"]) else metrics["auc"])

        else:

            scheduler.step()

        lrs = [g["lr"] for g in optimizer.param_groups]

        print(f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

        print(

            f"Val AUC: {metrics['auc']:.4f} | Acc: {metrics['acc']:.4f} | "

            f"Sens: {metrics['sens']:.4f} | Spec: {metrics['spec']:.4f} | Thr: {thr:.2f}"

        )

        print(f"LR(s): {', '.join(f'{lr:.2e}' for lr in lrs)}")

        # 保存条件：selection score 提升 > min_delta

        if score > best_score + config["min_delta"]:

            best_score = score

            patience_counter = 0

            torch.save({

                "model_state": model.state_dict(),

                "threshold": thr,          # 保存 best model 对应的阈值

                "metrics": metrics,

                "score": score,

                "epoch": epoch + 1,

                "config": config,

            }, best_path)

            print(f"--> Saved best model (score={score:.4f}, thr={thr:.2f})")

        else:

            patience_counter += 1

            print(f"No improvement ({patience_counter}/{config['patience']})")

            if patience_counter >= config["patience"]:

                print("Early stopping triggered.")

                break

    # ========== 最终评估 ==========

    print("\n" + "=" * 70)

    print("Final evaluation with best model")

    print("=" * 70)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])

    best_thr = ckpt["threshold"]   # 关键：用 best model 保存的阈值

    print(f"Best checkpoint: epoch={ckpt.get('epoch')}, score={ckpt.get('score'):.4f}, thr={best_thr:.2f}")

    # 内部验证

    val_metrics, val_raw, _ = evaluate_model(

        model, config["data_root"], "internal_val", config, is_fusion,

        criterion, device, best_thr, desc="Internal Val",

    )

    print(

        f"[Internal Val] AUC={val_metrics['auc']:.4f} Acc={val_metrics['acc']:.4f} "

        f"Sens={val_metrics['sens']:.4f} Spec={val_metrics['spec']:.4f} "

        f"Prec={val_metrics['precision']:.4f} F1={val_metrics['f1']:.4f} Thr={best_thr:.2f}"

    )

    save_predictions_csv(

        val_raw["patient_ids"], val_raw["labels"], val_raw["probs"], best_thr,

        os.path.join(config["save_dir"], "internal_val_predictions.csv"),

    )

    pd.DataFrame([{

        "split": "internal_val", "auc": val_metrics["auc"], "acc": val_metrics["acc"],

        "sens": val_metrics["sens"], "spec": val_metrics["spec"],

        "precision": val_metrics["precision"], "f1": val_metrics["f1"],

        "threshold": best_thr, "score": best_score,

    }]).to_csv(

        os.path.join(config["save_dir"], "internal_val_summary.csv"),

        index=False, encoding="utf-8-sig",

    )

    # 训练集

    train_metrics, train_raw, _ = evaluate_model(

        model, config["data_root"], "train", config, is_fusion,

        criterion, device, best_thr, desc="Train Eval",

    )

    print(

        f"[Train] AUC={train_metrics['auc']:.4f} Acc={train_metrics['acc']:.4f} "

        f"Sens={train_metrics['sens']:.4f} Spec={train_metrics['spec']:.4f} "

        f"Prec={train_metrics['precision']:.4f} F1={train_metrics['f1']:.4f} Thr={best_thr:.2f}"

    )

    save_predictions_csv(

        train_raw["patient_ids"], train_raw["labels"], train_raw["probs"], best_thr,

        os.path.join(config["save_dir"], "train_predictions.csv"),

    )

    pd.DataFrame([{

        "split": "train", "auc": train_metrics["auc"], "acc": train_metrics["acc"],

        "sens": train_metrics["sens"], "spec": train_metrics["spec"],

        "precision": train_metrics["precision"], "f1": train_metrics["f1"],

        "threshold": best_thr, "score": best_score,

    }]).to_csv(

        os.path.join(config["save_dir"], "train_summary.csv"),

        index=False, encoding="utf-8-sig",

    )

    # 外部验证

    ext_split = config.get("external_split")

    if ext_split and os.path.exists(os.path.join(config["data_root"], ext_split)):

        print("\n" + "=" * 70)

        print(f"External validation: {ext_split}")

        print("=" * 70)

        try:

            ext_metrics, ext_raw, ext_n = evaluate_model(

                model, config["data_root"], ext_split, config, is_fusion,

                criterion, device, best_thr, desc="External",

            )

            print(

                f"[External N={ext_n}] AUC={ext_metrics['auc']:.4f} Acc={ext_metrics['acc']:.4f} "

                f"Sens={ext_metrics['sens']:.4f} Spec={ext_metrics['spec']:.4f} "

                f"Prec={ext_metrics['precision']:.4f} F1={ext_metrics['f1']:.4f} Thr={best_thr:.2f}"

            )

            save_predictions_csv(

                ext_raw["patient_ids"], ext_raw["labels"], ext_raw["probs"], best_thr,

                os.path.join(config["save_dir"], "external_predictions.csv"),

            )

            pd.DataFrame([{

                "split": ext_split, "n": ext_n, "auc": ext_metrics["auc"],

                "acc": ext_metrics["acc"], "sens": ext_metrics["sens"],

                "spec": ext_metrics["spec"], "precision": ext_metrics["precision"],

                "f1": ext_metrics["f1"], "threshold": best_thr, "score": best_score,

            }]).to_csv(

                os.path.join(config["save_dir"], "external_summary.csv"),

                index=False, encoding="utf-8-sig",

            )

        except Exception as e:

            print(f"[Warning] External validation failed: {e}")

    else:

        print(f"\n[Skip] External split '{ext_split}' not found or not specified.")

    print("\nAll done.")

if __name__ == "__main__":

    config = {
        "data_root": "data/processed",
        "clinical_csv": "data/clinical_features_8lasso.csv",

        "save_dir": r"./checkpoints",

        "batch_size": 4,

        "num_epochs": 80,

        "lr": 1e-4,

        "weight_decay": 1e-4,

        "patience": 12,

        "min_delta": 0.001,

        "seed": 62,

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # 阈值候选网格：包含 0.65，permits 0.65 if selected

        "threshold_candidates": (0.40, 0.45, 0.50, 0.55, 0.60, 0.65),

        # 3D+clinical

        "clinical_dim": 8,

        "backbone_lr": 1e-5,

        "head_lr": 1e-4,

        "warmup_epochs": 5,

        "min_lr_ratio": 0.01,

        "external_split": "external_test",

        # Baseline:    use_optimized=False, is_fusion=False

        # Optimized:   use_optimized=True,  is_fusion=False

        # 3D+Clinical: is_fusion=True

        "use_optimized": True,

        "is_fusion": False,

    }

    # Command-line overrides only; model fitting/threshold selection are unchanged.
    parser = argparse.ArgumentParser(description="3D CrAg model training")
    parser.add_argument("--model", choices=["baseline", "optimized", "fusion"], default="optimized")
    parser.add_argument("--data-root", default=config["data_root"])
    parser.add_argument("--clinical-csv", default=config["clinical_csv"])
    parser.add_argument("--save-dir", default=config["save_dir"])
    parser.add_argument("--external-split", default=config["external_split"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    args = parser.parse_args()
    config.update(data_root=args.data_root, clinical_csv=args.clinical_csv,
                  save_dir=args.save_dir, external_split=args.external_split,
                  seed=args.seed)
    config["is_fusion"] = args.model == "fusion"
    config["use_optimized"] = args.model != "baseline"
    train_experiment(config)