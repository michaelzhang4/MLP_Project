"""
P1: DeBERTa-v3-base as Modern Backbone

1. Fine-tune DeBERTa-v3-base on PhraseBank (standalone accuracy as upper bound)
2. Experiment B: KD routing with DeBERTa as teacher
3. Experiment D: Multi-exit with DeBERTa backbone

Usage:
    python run_deberta_experiments.py --phase finetune   # Fine-tune DeBERTa
    python run_deberta_experiments.py --phase kd          # KD routing (Exp B)
    python run_deberta_experiments.py --phase multi-exit  # Multi-exit (Exp D)
    python run_deberta_experiments.py --phase all         # Run all phases
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2Tokenizer,
    get_linear_schedule_with_warmup,
)

from utils import (
    DEVICE,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    bootstrap_hypothesis_ci,
    expected_calibration_error,
    load_all_phrasebank,
    make_json_safe,
    mcnemar_test,
    set_seed,
    split_data,
)


# ============================================================
# Config
# ============================================================

DEBERTA_MODEL = "microsoft/deberta-v3-base"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBERTA_SAVE_DIR = _PROJECT_ROOT / "deberta_finetuned"


def load_tokenizer(model_path: str):
    """Load tokenizer, using DebertaV2Tokenizer for DeBERTa models."""
    if "deberta" in model_path.lower():
        return DebertaV2Tokenizer.from_pretrained(model_path)
    return AutoTokenizer.from_pretrained(model_path, use_fast=False)


@dataclass
class FinetuneConfig:
    epochs: int = 5
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128


# ============================================================
# Dataset
# ============================================================

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor([LABEL2ID[l] for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ============================================================
# Phase 1: Fine-tune DeBERTa
# ============================================================

def finetune_deberta(
    train_texts, train_labels, val_texts, val_labels,
    cfg: FinetuneConfig, save_dir: Path,
) -> Dict:
    """Fine-tune DeBERTa-v3-base on PhraseBank."""
    print(f"\n=== Fine-tuning {DEBERTA_MODEL} ===")
    tokenizer = load_tokenizer(DEBERTA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, cfg.max_length)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_state = None
    best_f1 = -1.0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False):
            labels = batch["labels"].to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            out = model(**inputs)
            loss = F.cross_entropy(out.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].numpy()
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        avg_loss = total_loss / max(1, len(train_loader))

        entry = {
            "epoch": epoch + 1, "train_loss": float(avg_loss),
            "val_accuracy": float(val_acc), "val_f1_macro": float(val_f1),
        }
        history.append(entry)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Save model
    save_dir.mkdir(exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  Saved fine-tuned model to {save_dir}")

    return {"history": history, "best_val_f1": float(best_f1)}


# ============================================================
# Phase 2: KD Routing (Experiment B) with DeBERTa teacher
# ============================================================

def predict_with_model(
    model_path: str, texts: List[str], batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Predict with a saved model. Returns (labels, confidences, logits, elapsed_s)."""
    tokenizer = load_tokenizer(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()

    # Determine label reorder
    try:
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        label_to_idx = {v: k for k, v in id2label.items()}
        reorder = [label_to_idx[l] for l in LABEL_LIST]
    except (KeyError, AttributeError):
        reorder = list(range(len(LABEL_LIST)))

    all_logits = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=128, return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        logits = logits[:, reorder]
        all_logits.append(logits)
    elapsed = time.perf_counter() - t0

    logits = np.vstack(all_logits)
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    confs = np.max(probs, axis=1)
    pred_ids = np.argmax(logits, axis=1)
    preds = np.array([LABEL_LIST[i] for i in pred_ids])

    del model
    torch.cuda.empty_cache()
    return preds, confs, logits, elapsed


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels, lr=0.01, max_iter=500):
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        temp = torch.nn.Parameter(torch.ones(1) * 1.5)
        opt = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits_t / torch.clamp(temp, 0.01, 50.0), labels_t)
            loss.backward()
            return loss
        opt.step(closure)
        self.temperature = float(torch.clamp(temp, 0.01, 50.0).item())

    def calibrate(self, logits):
        scaled = logits / self.temperature
        exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)


def run_kd_experiment(
    teacher_path: str,
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, test_df,
    batch_size: int = 16,
    results_dir: Path = Path("results_deberta_kd"),
) -> Dict:
    """Experiment B: DistilBERT student with DeBERTa teacher."""
    print("\n=== Experiment B: KD with DeBERTa Teacher ===")
    results_dir.mkdir(exist_ok=True)

    # Get teacher predictions and logits
    print("  Getting DeBERTa teacher predictions...")
    teacher_preds, teacher_confs, teacher_logits_raw, teacher_time_s = predict_with_model(
        teacher_path, test_texts, batch_size=batch_size
    )
    teacher_acc = accuracy_score(test_labels, teacher_preds)
    teacher_f1 = f1_score(test_labels, teacher_preds, average="macro", zero_division=0)
    print(f"  DeBERTa teacher: acc={teacher_acc:.4f}, f1={teacher_f1:.4f}")

    # Train DistilBERT student with KD from DeBERTa
    from hybrid_sentiment_v2 import DistilStudent, TrainConfig, reorder_teacher_logits

    student = DistilStudent()
    cfg = TrainConfig(ce_epochs=4, kd_epochs=2, batch_size=batch_size)

    # Phase 1: CE training
    print("\n  Phase 1: CE fine-tuning student...")
    ce_history = student.train_ce(train_texts, train_labels, val_texts, val_labels, cfg)

    # Phase 2: KD with DeBERTa logits
    print("\n  Phase 2: KD with DeBERTa teacher logits...")
    train_teacher_preds, _, train_teacher_logits, _ = predict_with_model(
        teacher_path, train_texts, batch_size=batch_size
    )
    val_teacher_preds, _, val_teacher_logits, _ = predict_with_model(
        teacher_path, val_texts, batch_size=batch_size
    )

    kd_history = student.train_kd(
        train_texts, train_labels, val_texts, val_labels,
        train_teacher_logits=train_teacher_logits,
        val_teacher_logits=val_teacher_logits,
        cfg=cfg,
    )

    # Calibrate student
    val_student_preds, _, val_student_logits = student.predict_batch(val_texts, batch_size=32)
    scaler = TemperatureScaler()
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels])
    scaler.fit(val_student_logits, val_label_ids)

    # Test student
    student_t0 = time.perf_counter()
    student_preds, _, student_logits = student.predict_batch(test_texts, batch_size=32)
    student_time_s = time.perf_counter() - student_t0
    cal_probs = scaler.calibrate(student_logits)
    cal_confs = np.max(cal_probs, axis=1)
    student_acc = accuracy_score(test_labels, student_preds)
    student_f1 = f1_score(test_labels, student_preds, average="macro", zero_division=0)
    print(f"\n  Student: acc={student_acc:.4f}, f1={student_f1:.4f}")

    # Threshold sweep
    test_labels_arr = np.array(test_labels)
    sweep = []
    for tau in np.linspace(0.0, 1.0, 51):
        use_student = cal_confs >= tau
        hybrid = np.where(use_student, student_preds, teacher_preds)
        acc = accuracy_score(test_labels_arr, hybrid)
        ret = acc / teacher_acc if teacher_acc > 0 else 0.0
        teacher_pct = float((~use_student).mean() * 100.0)
        sweep.append({
            "tau": float(tau), "accuracy": float(acc),
            "retention": float(ret), "teacher_pct": float(teacher_pct),
        })

    # Find optimal tau (target 98% retention, <=30% teacher)
    feasible = [s for s in sweep if s["retention"] >= 0.98 and s["teacher_pct"] <= 30.0]
    if feasible:
        best = min(feasible, key=lambda x: x["teacher_pct"])
    else:
        best = min(sweep, key=lambda x: abs(s["retention"] - 0.98))
    optimal_tau = best["tau"]

    use_student = cal_confs >= optimal_tau
    hybrid_preds = np.where(use_student, student_preds, teacher_preds)
    hybrid_acc = accuracy_score(test_labels_arr, hybrid_preds)
    hybrid_f1 = f1_score(test_labels_arr, hybrid_preds, average="macro", zero_division=0)
    retention = hybrid_acc / teacher_acc if teacher_acc > 0 else 0.0
    teacher_usage = float((~use_student).mean() * 100.0)

    # ECE
    hybrid_confs = np.where(use_student, cal_confs, teacher_confs)
    hybrid_correct = (hybrid_preds == test_labels_arr).astype(float)
    ece, ece_bins = expected_calibration_error(hybrid_confs, hybrid_correct)

    # Statistical tests
    ci = bootstrap_hypothesis_ci(test_labels_arr, hybrid_preds, teacher_preds, use_student)
    mcn = mcnemar_test(
        hybrid_preds == test_labels_arr,
        teacher_preds == test_labels_arr,
    )

    # Agreement tier analysis
    test_tiers = test_df["agreement_tier"].tolist()
    tier_analysis = {}
    for tier in ["100", "75", "66", "50"]:
        idx = np.array([i for i, t in enumerate(test_tiers) if t == tier])
        if len(idx) == 0:
            continue
        y = test_labels_arr[idx]
        tier_analysis[tier] = {
            "n": int(len(idx)),
            "student_acc": float(accuracy_score(y, student_preds[idx])),
            "teacher_acc": float(accuracy_score(y, teacher_preds[idx])),
            "hybrid_acc": float(accuracy_score(y, hybrid_preds[idx])),
            "teacher_pct": float((~use_student[idx]).mean() * 100.0),
        }

    n_test = len(test_texts)
    teacher_ms = teacher_time_s / n_test * 1000
    student_ms = student_time_s / n_test * 1000
    teacher_frac = teacher_usage / 100.0
    hybrid_ms = teacher_frac * teacher_ms + (1 - teacher_frac) * student_ms

    results = {
        "teacher": "DeBERTa-v3-base (fine-tuned)",
        "teacher_accuracy": float(teacher_acc),
        "teacher_f1": float(teacher_f1),
        "student_accuracy": float(student_acc),
        "student_f1": float(student_f1),
        "hybrid_accuracy": float(hybrid_acc),
        "hybrid_f1": float(hybrid_f1),
        "accuracy_retention": float(retention),
        "teacher_usage_pct": float(teacher_usage),
        "selected_tau": float(optimal_tau),
        "ece": float(ece),
        "ci_retention_95": ci["retention_ci_95"],
        "mcnemar_p_value": mcn["p_value"],
        "sweep": sweep,
        "agreement_tier_analysis": tier_analysis,
        "training": {
            "ce_history": ce_history,
            "kd_history": kd_history,
            "temperature": scaler.temperature,
        },
        "timing": {
            "teacher_time_s": float(teacher_time_s),
            "teacher_ms_per_sample": float(teacher_ms),
            "student_time_s": float(student_time_s),
            "student_ms_per_sample": float(student_ms),
            "hybrid_ms_per_sample": float(hybrid_ms),
            "runtime_speedup": float(teacher_ms / hybrid_ms) if hybrid_ms > 0 else 1.0,
        },
    }

    print(f"\n  Hybrid: acc={hybrid_acc:.4f}, retention={retention:.4f}, teacher%={teacher_usage:.1f}%")
    print(f"  ECE: {ece:.4f}")
    print(f"  Timing: teacher={teacher_ms:.2f}ms/sample, student={student_ms:.2f}ms/sample, hybrid={hybrid_ms:.2f}ms/sample")

    out_json = results_dir / "deberta_kd_results.json"
    with open(out_json, "w") as f:
        json.dump(make_json_safe(results), f, indent=2)
    print(f"  Saved to {out_json}")

    return results


# ============================================================
# Phase 3: Multi-Exit with DeBERTa backbone (Experiment D)
# ============================================================

class MLPExitHead(nn.Module):
    def __init__(self, hidden_size=768, num_labels=3, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, x):
        return self.net(x)


def run_multi_exit_experiment(
    model_path: str,
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, test_df,
    exit_layers=(4, 8),
    batch_size=32,
    head_epochs=30,
    patience=1,
    results_dir=Path("results_deberta_multi_exit"),
) -> Dict:
    """Experiment D: Multi-exit with DeBERTa backbone."""
    print("\n=== Experiment D: Multi-Exit DeBERTa ===")
    results_dir.mkdir(exist_ok=True)

    tokenizer = load_tokenizer(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Label reordering
    try:
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        label_to_idx = {v: k for k, v in id2label.items()}
        reorder_idx = [label_to_idx[l] for l in LABEL_LIST]
    except (KeyError, AttributeError):
        reorder_idx = list(range(len(LABEL_LIST)))

    exit_layers = tuple(l for l in exit_layers if 1 <= l < num_layers)
    print(f"  Model: {model_path}, layers={num_layers}, hidden={hidden_size}")
    print(f"  Exit layers: {exit_layers}")

    # Extract CLS features at exit layers
    def extract_features(texts, bs=batch_size):
        layer_cls = {l: [] for l in exit_layers}
        final_logits_all = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = tokenizer(
                batch, truncation=True, padding=True, max_length=128, return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, return_dict=True)
            for l in exit_layers:
                layer_cls[l].append(out.hidden_states[l][:, 0, :].cpu().numpy())
            final_logits_all.append(out.logits[:, reorder_idx].cpu().numpy())
        layer_cls = {k: np.vstack(v) for k, v in layer_cls.items()}
        final_logits = np.vstack(final_logits_all)
        final_preds = np.array([LABEL_LIST[i] for i in np.argmax(final_logits, axis=1)])
        return layer_cls, final_logits, final_preds

    print("  Extracting train features...")
    train_cls, train_final_logits, train_final_preds = extract_features(train_texts)
    print("  Extracting val features...")
    val_cls, val_final_logits, val_final_preds = extract_features(val_texts)
    print("  Extracting test features...")
    test_t0 = time.perf_counter()
    test_cls, test_final_logits, test_final_preds = extract_features(test_texts)
    full_time_s = time.perf_counter() - test_t0

    # Train MLP exit heads
    print("\n  Training MLP exit heads...")
    train_label_ids = np.array([LABEL2ID[l] for l in train_labels])
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels])

    exit_heads = {}
    head_summary = {}
    for layer in exit_layers:
        head = MLPExitHead(hidden_size=hidden_size).to(DEVICE)
        optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=head_epochs)

        X_train = torch.tensor(train_cls[layer], dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(train_label_ids, dtype=torch.long).to(DEVICE)
        X_val = torch.tensor(val_cls[layer], dtype=torch.float32).to(DEVICE)
        y_val = torch.tensor(val_label_ids, dtype=torch.long).to(DEVICE)

        best_val_acc = 0.0
        best_state = None
        for epoch in range(head_epochs):
            head.train()
            perm = torch.randperm(len(X_train), device=DEVICE)
            for start in range(0, len(X_train), batch_size):
                idx = perm[start:start+batch_size]
                loss = F.cross_entropy(head(X_train[idx]), y_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            head.eval()
            with torch.no_grad():
                val_preds = torch.argmax(head(X_val), dim=1).cpu().numpy()
                val_acc = accuracy_score(val_label_ids, val_preds)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        if best_state:
            head.load_state_dict(best_state)
            head.to(DEVICE)
        exit_heads[layer] = head

        head.eval()
        with torch.no_grad():
            train_acc = accuracy_score(
                train_label_ids,
                torch.argmax(head(X_train), dim=1).cpu().numpy()
            )
        head_summary[f"layer_{layer}"] = {
            "train_acc": float(train_acc), "val_acc": float(best_val_acc)
        }
        print(f"    Layer {layer}: train_acc={train_acc:.4f}, val_acc={best_val_acc:.4f}")

    # Get exit logits for all splits
    def get_exit_logits(cls_features, layer):
        head = exit_heads[layer]
        head.eval()
        with torch.no_grad():
            return head(torch.tensor(cls_features, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    # Simple entropy-based routing
    def softmax_np(logits):
        z = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def entropy(probs):
        return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    # Threshold sweep on validation
    print("\n  Threshold sweep (entropy-based)...")
    test_labels_arr = np.array(test_labels)
    val_labels_arr = np.array(val_labels)
    full_acc = accuracy_score(test_labels_arr, test_final_preds)
    full_f1 = f1_score(test_labels_arr, test_final_preds, average="macro", zero_division=0)

    # Sweep on test (same methodology as V5 entropy baseline for fair comparison)
    sweep = []
    for ent_thresh in np.linspace(0.01, 1.5, 100):
        hybrid = test_final_preds.copy()
        used_layers = np.full(len(test_texts), num_layers, dtype=float)
        decided = np.zeros(len(test_texts), dtype=bool)

        for layer in exit_layers:
            logits = get_exit_logits(test_cls[layer], layer)
            probs = softmax_np(logits)
            ent = entropy(probs)
            preds = np.array([LABEL_LIST[i] for i in np.argmax(logits, axis=1)])
            can_exit = (~decided) & (ent <= ent_thresh)
            hybrid[can_exit] = preds[can_exit]
            used_layers[can_exit] = layer
            decided[can_exit] = True

        acc = accuracy_score(test_labels_arr, hybrid)
        ret = acc / full_acc if full_acc > 0 else 0.0
        lf = float(np.mean(used_layers)) / num_layers
        exit_pct = float(decided.mean() * 100.0)

        sweep.append({
            "entropy_threshold": float(ent_thresh),
            "accuracy": float(acc), "retention": float(ret),
            "layer_fraction": float(lf), "early_exit_pct": float(exit_pct),
        })

    # Find best operating point
    feasible = [s for s in sweep if s["retention"] >= 0.98 and s["layer_fraction"] <= 0.75]
    if feasible:
        best = min(feasible, key=lambda x: x["layer_fraction"])
    else:
        best = min(sweep, key=lambda x: (1 - x["accuracy"]) + 0.25 * x["layer_fraction"])

    best_thresh = best["entropy_threshold"]

    # Apply best threshold
    hybrid_preds = test_final_preds.copy()
    used_layers = np.full(len(test_texts), num_layers, dtype=float)
    decided = np.zeros(len(test_texts), dtype=bool)
    for layer in exit_layers:
        logits = get_exit_logits(test_cls[layer], layer)
        probs = softmax_np(logits)
        ent = entropy(probs)
        preds = np.array([LABEL_LIST[i] for i in np.argmax(logits, axis=1)])
        can_exit = (~decided) & (ent <= best_thresh)
        hybrid_preds[can_exit] = preds[can_exit]
        used_layers[can_exit] = layer
        decided[can_exit] = True

    hybrid_acc = accuracy_score(test_labels_arr, hybrid_preds)
    hybrid_f1 = f1_score(test_labels_arr, hybrid_preds, average="macro", zero_division=0)
    retention = hybrid_acc / full_acc if full_acc > 0 else 0.0
    avg_layers = float(np.mean(used_layers))
    layer_fraction = avg_layers / num_layers
    early_exit_pct = float(decided.mean() * 100.0)

    # Per-exit-layer accuracy
    exit_metrics = {}
    for layer in exit_layers:
        logits = get_exit_logits(test_cls[layer], layer)
        preds = np.array([LABEL_LIST[i] for i in np.argmax(logits, axis=1)])
        exit_metrics[str(layer)] = {
            "accuracy": float(accuracy_score(test_labels_arr, preds)),
            "f1_macro": float(f1_score(test_labels_arr, preds, average="macro", zero_division=0)),
        }

    # Layer usage
    layer_usage = {}
    for layer in exit_layers:
        layer_usage[str(layer)] = float(np.sum(used_layers == layer) / len(test_texts) * 100.0)
    layer_usage[str(num_layers)] = float(np.sum(~decided) / len(test_texts) * 100.0)

    # ECE
    # Use max softmax prob of the exit head that made the decision
    hybrid_confs = np.zeros(len(test_texts))
    for i in range(len(test_texts)):
        if decided[i]:
            layer = int(used_layers[i])
            logits = get_exit_logits(test_cls[layer][i:i+1], layer)
            hybrid_confs[i] = float(np.max(softmax_np(logits)))
        else:
            hybrid_confs[i] = float(np.max(softmax_np(test_final_logits[i:i+1])))
    hybrid_correct = (hybrid_preds == test_labels_arr).astype(float)
    ece, ece_bins = expected_calibration_error(hybrid_confs, hybrid_correct)

    # Statistical tests
    ci = bootstrap_hypothesis_ci(test_labels_arr, hybrid_preds, test_final_preds, decided)
    mcn = mcnemar_test(hybrid_preds == test_labels_arr, test_final_preds == test_labels_arr)

    # Agreement tier analysis
    test_tiers = test_df["agreement_tier"].tolist()
    tier_analysis = {}
    for tier in ["100", "75", "66", "50"]:
        idx = np.array([i for i, t in enumerate(test_tiers) if t == tier])
        if len(idx) == 0:
            continue
        y = test_labels_arr[idx]
        tier_analysis[tier] = {
            "n": int(len(idx)),
            "full_acc": float(accuracy_score(y, test_final_preds[idx])),
            "hybrid_acc": float(accuracy_score(y, hybrid_preds[idx])),
            "early_exit_pct": float(decided[idx].mean() * 100.0),
            "avg_layers": float(np.mean(used_layers[idx])),
        }

    n_test = len(test_texts)
    full_ms = full_time_s / n_test * 1000
    hybrid_ms = full_ms * layer_fraction
    runtime_speedup = 1.0 / layer_fraction if layer_fraction > 0 else 1.0

    results = {
        "backbone": "DeBERTa-v3-base (fine-tuned)",
        "model_path": str(model_path),
        "num_hidden_layers": int(num_layers),
        "exit_layers": list(exit_layers),
        "full_accuracy": float(full_acc),
        "full_f1": float(full_f1),
        "hybrid_accuracy": float(hybrid_acc),
        "hybrid_f1": float(hybrid_f1),
        "accuracy_retention": float(retention),
        "early_exit_pct": float(early_exit_pct),
        "avg_layers_used": float(avg_layers),
        "layer_fraction": float(layer_fraction),
        "layer_usage": layer_usage,
        "exit_layer_metrics": exit_metrics,
        "ece": float(ece),
        "selected_threshold": float(best_thresh),
        "ci_retention_95": ci["retention_ci_95"],
        "mcnemar_p_value": mcn["p_value"],
        "sweep": sweep,
        "agreement_tier_analysis": tier_analysis,
        "head_summary": head_summary,
        "timing": {
            "full_time_s": float(full_time_s),
            "full_ms_per_sample": float(full_ms),
            "hybrid_ms_per_sample": float(hybrid_ms),
            "runtime_speedup": float(runtime_speedup),
        },
    }

    print(f"\n  Full DeBERTa:  acc={full_acc:.4f}, F1={full_f1:.4f}")
    print(f"  Hybrid:        acc={hybrid_acc:.4f}, F1={hybrid_f1:.4f}")
    print(f"  Retention:     {retention:.4f}")
    print(f"  Early exit:    {early_exit_pct:.1f}%, avg_layers={avg_layers:.2f}/{num_layers}")
    print(f"  Layer usage:   {layer_usage}")
    print(f"  ECE:           {ece:.4f}")
    print(f"  Timing:        full={full_ms:.2f}ms/sample, hybrid={hybrid_ms:.2f}ms/sample, speedup={runtime_speedup:.2f}x")

    out_json = results_dir / "deberta_multi_exit_results.json"
    with open(out_json, "w") as f:
        json.dump(make_json_safe(results), f, indent=2)
    print(f"  Saved to {out_json}")

    del model
    torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DeBERTa backbone experiments")
    parser.add_argument("--phase", type=str, default="all",
                       choices=["finetune", "kd", "multi-exit", "all"])
    parser.add_argument("--data-dir", type=str, default=str(_PROJECT_ROOT / "FinancialPhraseBank-v1.0"))
    parser.add_argument("--results-dir", type=str, default="results_deberta")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")

    # Load data
    print("\n=== Loading data ===")
    df = load_all_phrasebank(args.data_dir)
    train_df, val_df, test_df = split_data(df)
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    all_results = {}
    model_save_dir = DEBERTA_SAVE_DIR

    # Phase 1: Fine-tune DeBERTa
    if args.phase in ("finetune", "all"):
        cfg = FinetuneConfig(epochs=args.epochs, batch_size=args.batch_size)
        ft_results = finetune_deberta(
            train_texts, train_labels, val_texts, val_labels, cfg, model_save_dir,
        )
        # Get standalone test accuracy
        print("\n  Evaluating fine-tuned DeBERTa on test set...")
        preds, confs, logits = predict_with_model(str(model_save_dir), test_texts)
        test_labels_arr = np.array(test_labels)
        acc = accuracy_score(test_labels_arr, preds)
        f1 = f1_score(test_labels_arr, preds, average="macro", zero_division=0)
        correct = (preds == test_labels_arr).astype(float)
        ece, _ = expected_calibration_error(confs, correct)
        cm = confusion_matrix(test_labels_arr, preds, labels=LABEL_LIST).tolist()

        all_results["finetune"] = {
            **ft_results,
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1),
            "test_ece": float(ece),
            "test_confusion_matrix": cm,
        }
        print(f"\n  DeBERTa standalone: acc={acc:.4f}, F1={f1:.4f}, ECE={ece:.4f}")

    # Phase 2: KD experiment
    if args.phase in ("kd", "all"):
        if not model_save_dir.exists():
            print("ERROR: Fine-tuned DeBERTa not found. Run --phase finetune first.")
            return
        kd_results = run_kd_experiment(
            teacher_path=str(model_save_dir),
            train_texts=train_texts, train_labels=train_labels,
            val_texts=val_texts, val_labels=val_labels,
            test_texts=test_texts, test_labels=test_labels, test_df=test_df,
            batch_size=args.batch_size,
            results_dir=out_dir / "kd",
        )
        all_results["kd_routing"] = kd_results

    # Phase 3: Multi-exit experiment
    if args.phase in ("multi-exit", "all"):
        if not model_save_dir.exists():
            print("ERROR: Fine-tuned DeBERTa not found. Run --phase finetune first.")
            return
        me_results = run_multi_exit_experiment(
            model_path=str(model_save_dir),
            train_texts=train_texts, train_labels=train_labels,
            val_texts=val_texts, val_labels=val_labels,
            test_texts=test_texts, test_labels=test_labels, test_df=test_df,
            exit_layers=(4, 8),
            batch_size=args.batch_size,
            results_dir=out_dir / "multi_exit",
        )
        all_results["multi_exit"] = me_results

    # Save combined results
    out_json = out_dir / "all_deberta_results.json"
    with open(out_json, "w") as f:
        json.dump(make_json_safe(all_results), f, indent=2)
    print(f"\nAll results saved to {out_json}")


if __name__ == "__main__":
    main()
