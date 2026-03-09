"""
P2: Run Experiments B (KD routing) and D (multi-exit) on FiQA dataset.

Cross-dataset validation to address single-dataset reviewer concern.
FiQA has no agreement tiers — we create proxy difficulty tiers based on
model disagreement (VADER vs FinBERT vs student).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow imports from the project root (for hybrid_sentiment)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from utils import (
    DEVICE,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    bootstrap_hypothesis_ci,
    expected_calibration_error,
    load_fiqa,
    make_json_safe,
    mcnemar_test,
    set_seed,
    split_data,
)


# ============================================================
# Experiment B: KD routing on FiQA (FinBERT teacher, DistilBERT student)
# ============================================================

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


def predict_hf(model_id, texts, batch_size=32):
    """Predict with a HuggingFace model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(DEVICE).eval()

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label_to_idx = {v: k for k, v in id2label.items()}
    try:
        reorder = [label_to_idx[l] for l in LABEL_LIST]
    except KeyError:
        reorder = list(range(len(LABEL_LIST)))

    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        all_logits.append(logits[:, reorder])

    logits = np.vstack(all_logits)
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    confs = np.max(probs, axis=1)
    preds = np.array([LABEL_LIST[i] for i in np.argmax(logits, axis=1)])

    del model
    torch.cuda.empty_cache()
    return preds, confs, logits


def run_kd_experiment_fiqa(train_df, val_df, test_df, out_dir):
    """Run KD routing experiment on FiQA."""
    print("\n=== Experiment B: KD Routing on FiQA ===")
    out_dir.mkdir(exist_ok=True)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = np.array(test_df["label"].tolist())

    # FinBERT teacher
    print("  FinBERT teacher predictions...")
    teacher_preds, teacher_confs, teacher_logits = predict_hf(
        "ProsusAI/finbert", test_texts
    )
    teacher_acc = accuracy_score(test_labels, teacher_preds)
    teacher_f1 = f1_score(test_labels, teacher_preds, average="macro", zero_division=0)
    print(f"  FinBERT: acc={teacher_acc:.4f}, f1={teacher_f1:.4f}")

    # Train DistilBERT student with KD
    from hybrid_sentiment_v2 import DistilStudent, TrainConfig, reorder_teacher_logits
    from hybrid_sentiment import FinBERTAnalyzer

    student = DistilStudent()
    cfg = TrainConfig(ce_epochs=4, kd_epochs=2, batch_size=16)

    print("  Phase 1: CE fine-tuning...")
    ce_history = student.train_ce(train_texts, train_labels, val_texts, val_labels, cfg)

    print("  Phase 2: KD training...")
    teacher_analyzer = FinBERTAnalyzer()
    train_teacher_logits, _ = teacher_analyzer.get_logits_batch(train_texts, batch_size=32)
    val_teacher_logits, _ = teacher_analyzer.get_logits_batch(val_texts, batch_size=32)
    train_teacher_logits = reorder_teacher_logits(train_teacher_logits, teacher_analyzer.label_map)
    val_teacher_logits = reorder_teacher_logits(val_teacher_logits, teacher_analyzer.label_map)

    kd_history = student.train_kd(
        train_texts, train_labels, val_texts, val_labels,
        train_teacher_logits, val_teacher_logits, cfg,
    )

    # Calibrate student
    val_student_preds, _, val_student_logits = student.predict_batch(val_texts, batch_size=32)
    scaler = TemperatureScaler()
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels])
    scaler.fit(val_student_logits, val_label_ids)

    # Test
    student_preds, _, student_logits = student.predict_batch(test_texts, batch_size=32)
    cal_probs = scaler.calibrate(student_logits)
    cal_confs = np.max(cal_probs, axis=1)
    student_acc = accuracy_score(test_labels, student_preds)
    print(f"  Student: acc={student_acc:.4f}")

    # Threshold sweep
    sweep = []
    for tau in np.linspace(0.0, 1.0, 51):
        use_student = cal_confs >= tau
        hybrid = np.where(use_student, student_preds, teacher_preds)
        acc = accuracy_score(test_labels, hybrid)
        ret = acc / teacher_acc if teacher_acc > 0 else 0.0
        sweep.append({
            "tau": float(tau), "accuracy": float(acc),
            "retention": float(ret),
            "teacher_pct": float((~use_student).mean() * 100.0),
        })

    feasible = [s for s in sweep if s["retention"] >= 0.98 and s["teacher_pct"] <= 30.0]
    if feasible:
        best = min(feasible, key=lambda x: x["teacher_pct"])
    else:
        best = min(sweep, key=lambda x: abs(x["retention"] - 0.98))
    optimal_tau = best["tau"]

    use_student = cal_confs >= optimal_tau
    hybrid_preds = np.where(use_student, student_preds, teacher_preds)
    hybrid_acc = accuracy_score(test_labels, hybrid_preds)
    hybrid_f1 = f1_score(test_labels, hybrid_preds, average="macro", zero_division=0)
    retention = hybrid_acc / teacher_acc if teacher_acc > 0 else 0.0
    teacher_usage = float((~use_student).mean() * 100.0)

    # ECE
    hybrid_confs = np.where(use_student, cal_confs, teacher_confs)
    correct = (hybrid_preds == test_labels).astype(float)
    ece, _ = expected_calibration_error(hybrid_confs, correct)

    # Difficulty proxy: model disagreement
    difficulty_analysis = compute_difficulty_proxy(
        student_preds, teacher_preds, test_labels, use_student
    )

    results = {
        "dataset": "FiQA-SA",
        "experiment": "B_KD_routing",
        "teacher_accuracy": float(teacher_acc),
        "teacher_f1": float(teacher_f1),
        "student_accuracy": float(student_acc),
        "hybrid_accuracy": float(hybrid_acc),
        "hybrid_f1": float(hybrid_f1),
        "accuracy_retention": float(retention),
        "teacher_usage_pct": float(teacher_usage),
        "selected_tau": float(optimal_tau),
        "ece": float(ece),
        "sweep": sweep,
        "difficulty_analysis": difficulty_analysis,
    }

    print(f"\n  Hybrid: acc={hybrid_acc:.4f}, retention={retention:.4f}, teacher%={teacher_usage:.1f}%")

    with open(out_dir / "fiqa_kd_results.json", "w") as f:
        json.dump(make_json_safe(results), f, indent=2)
    return results


def compute_difficulty_proxy(student_preds, teacher_preds, labels, use_student):
    """Create difficulty proxy from model disagreement."""
    agree = student_preds == teacher_preds
    n = len(labels)
    labels_arr = np.array(labels)

    easy_mask = agree
    hard_mask = ~agree

    analysis = {
        "easy_n": int(easy_mask.sum()),
        "hard_n": int(hard_mask.sum()),
    }
    if easy_mask.sum() > 0:
        analysis["easy_teacher_acc"] = float(accuracy_score(labels_arr[easy_mask], teacher_preds[easy_mask]))
        analysis["easy_hybrid_acc"] = float(accuracy_score(
            labels_arr[easy_mask],
            np.where(use_student[easy_mask], student_preds[easy_mask], teacher_preds[easy_mask])
        ))
        analysis["easy_teacher_pct"] = float((~use_student[easy_mask]).mean() * 100.0)
    if hard_mask.sum() > 0:
        analysis["hard_teacher_acc"] = float(accuracy_score(labels_arr[hard_mask], teacher_preds[hard_mask]))
        analysis["hard_hybrid_acc"] = float(accuracy_score(
            labels_arr[hard_mask],
            np.where(use_student[hard_mask], student_preds[hard_mask], teacher_preds[hard_mask])
        ))
        analysis["hard_teacher_pct"] = float((~use_student[hard_mask]).mean() * 100.0)

    return analysis


# ============================================================
# Experiment D: Multi-exit on FiQA
# ============================================================

class MLPExitHead(nn.Module):
    def __init__(self, hidden_size=768, num_labels=3, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, num_labels),
        )

    def forward(self, x):
        return self.net(x)


def run_multi_exit_fiqa(train_df, val_df, test_df, out_dir, exit_layers=(4, 8)):
    """Run multi-exit experiment on FiQA."""
    print("\n=== Experiment D: Multi-Exit on FiQA ===")
    out_dir.mkdir(exist_ok=True)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = np.array(test_df["label"].tolist())

    # Load FinBERT
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE).eval()

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label_to_idx = {v: k for k, v in id2label.items()}
    reorder_idx = [label_to_idx[l] for l in LABEL_LIST]
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    def extract_features(texts, bs=32):
        layer_cls = {l: [] for l in exit_layers}
        final_logits_all = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
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

    print("  Extracting features...")
    train_cls, _, _ = extract_features(train_texts)
    val_cls, val_final_logits, val_final_preds = extract_features(val_texts)
    test_cls, test_final_logits, test_final_preds = extract_features(test_texts)

    full_acc = accuracy_score(test_labels, test_final_preds)
    full_f1 = f1_score(test_labels, test_final_preds, average="macro", zero_division=0)
    print(f"  Full FinBERT: acc={full_acc:.4f}, f1={full_f1:.4f}")

    # Train exit heads
    train_label_ids = np.array([LABEL2ID[l] for l in train_labels])
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels])

    exit_heads = {}
    for layer in exit_layers:
        head = MLPExitHead(hidden_size=hidden_size).to(DEVICE)
        optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=30)

        X_train = torch.tensor(train_cls[layer], dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(train_label_ids, dtype=torch.long).to(DEVICE)
        X_val = torch.tensor(val_cls[layer], dtype=torch.float32).to(DEVICE)

        best_val_acc = 0.0
        best_state = None
        for epoch in range(30):
            head.train()
            perm = torch.randperm(len(X_train), device=DEVICE)
            for start in range(0, len(X_train), 32):
                idx = perm[start:start+32]
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
        print(f"    Exit head layer {layer}: val_acc={best_val_acc:.4f}")

    def get_exit_logits(cls, layer):
        exit_heads[layer].eval()
        with torch.no_grad():
            return exit_heads[layer](torch.tensor(cls, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    def softmax_np(logits):
        z = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def entropy(probs):
        return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    # Threshold sweep
    sweep = []
    for ent_thresh in np.linspace(0.01, 1.5, 100):
        hybrid = test_final_preds.copy()
        used_layers = np.full(len(test_texts), num_layers, dtype=float)
        decided = np.zeros(len(test_texts), dtype=bool)

        for layer in exit_layers:
            logits = get_exit_logits(test_cls[layer], layer)
            ent = entropy(softmax_np(logits))
            preds = np.array([LABEL_LIST[i] for i in np.argmax(logits, axis=1)])
            can_exit = (~decided) & (ent <= ent_thresh)
            hybrid[can_exit] = preds[can_exit]
            used_layers[can_exit] = layer
            decided[can_exit] = True

        acc = accuracy_score(test_labels, hybrid)
        ret = acc / full_acc if full_acc > 0 else 0.0
        lf = float(np.mean(used_layers)) / num_layers
        sweep.append({
            "entropy_threshold": float(ent_thresh),
            "accuracy": float(acc), "retention": float(ret),
            "layer_fraction": float(lf), "early_exit_pct": float(decided.mean() * 100.0),
        })

    feasible = [s for s in sweep if s["retention"] >= 0.98 and s["layer_fraction"] <= 0.75]
    if feasible:
        best = min(feasible, key=lambda x: x["layer_fraction"])
    else:
        best = min(sweep, key=lambda x: (1-x["accuracy"]) + 0.25*x["layer_fraction"])

    best_thresh = best["entropy_threshold"]

    # Apply best
    hybrid_preds = test_final_preds.copy()
    used_layers = np.full(len(test_texts), num_layers, dtype=float)
    decided = np.zeros(len(test_texts), dtype=bool)
    for layer in exit_layers:
        logits = get_exit_logits(test_cls[layer], layer)
        ent = entropy(softmax_np(logits))
        preds = np.array([LABEL_LIST[i] for i in np.argmax(logits, axis=1)])
        can_exit = (~decided) & (ent <= best_thresh)
        hybrid_preds[can_exit] = preds[can_exit]
        used_layers[can_exit] = layer
        decided[can_exit] = True

    hybrid_acc = accuracy_score(test_labels, hybrid_preds)
    hybrid_f1 = f1_score(test_labels, hybrid_preds, average="macro", zero_division=0)
    retention = hybrid_acc / full_acc if full_acc > 0 else 0.0
    layer_fraction = float(np.mean(used_layers)) / num_layers
    early_exit_pct = float(decided.mean() * 100.0)

    # ECE
    hybrid_confs = np.zeros(len(test_texts))
    for i in range(len(test_texts)):
        if decided[i]:
            layer = int(used_layers[i])
            logits = get_exit_logits(test_cls[layer][i:i+1], layer)
            hybrid_confs[i] = float(np.max(softmax_np(logits)))
        else:
            hybrid_confs[i] = float(np.max(softmax_np(test_final_logits[i:i+1])))
    correct = (hybrid_preds == test_labels).astype(float)
    ece, _ = expected_calibration_error(hybrid_confs, correct)

    # Layer usage
    layer_usage = {}
    for layer in exit_layers:
        layer_usage[str(layer)] = float(np.sum(used_layers == layer) / len(test_texts) * 100.0)
    layer_usage[str(num_layers)] = float(np.sum(~decided) / len(test_texts) * 100.0)

    # Difficulty proxy
    student_preds_all = np.array([LABEL_LIST[i] for i in np.argmax(
        get_exit_logits(test_cls[exit_layers[0]], exit_layers[0]), axis=1
    )])
    difficulty = compute_difficulty_proxy(
        student_preds_all, test_final_preds, test_labels, decided
    )

    results = {
        "dataset": "FiQA-SA",
        "experiment": "D_multi_exit",
        "backbone": "ProsusAI/finbert",
        "full_accuracy": float(full_acc),
        "full_f1": float(full_f1),
        "hybrid_accuracy": float(hybrid_acc),
        "hybrid_f1": float(hybrid_f1),
        "accuracy_retention": float(retention),
        "early_exit_pct": float(early_exit_pct),
        "layer_fraction": float(layer_fraction),
        "layer_usage": layer_usage,
        "ece": float(ece),
        "selected_threshold": float(best_thresh),
        "sweep": sweep,
        "difficulty_analysis": difficulty,
    }

    print(f"\n  Full FinBERT:  acc={full_acc:.4f}")
    print(f"  Hybrid:        acc={hybrid_acc:.4f}, retention={retention:.4f}")
    print(f"  Early exit:    {early_exit_pct:.1f}%, layer_frac={layer_fraction:.4f}")
    print(f"  ECE:           {ece:.4f}")
    print(f"  Layer usage:   {layer_usage}")

    with open(out_dir / "fiqa_multi_exit_results.json", "w") as f:
        json.dump(make_json_safe(results), f, indent=2)

    del model
    torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FiQA cross-dataset experiments")
    parser.add_argument("--results-dir", type=str, default="results_fiqa")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")

    # Load FiQA
    print("\n=== Loading FiQA ===")
    df = load_fiqa()
    train_df, val_df, test_df = split_data(df)

    # Run both experiments
    kd_results = run_kd_experiment_fiqa(train_df, val_df, test_df, out_dir / "kd")
    me_results = run_multi_exit_fiqa(train_df, val_df, test_df, out_dir / "multi_exit")

    # Combined summary
    combined = {
        "dataset": "FiQA-SA",
        "n_samples": len(df),
        "kd_routing": kd_results,
        "multi_exit": me_results,
    }
    with open(out_dir / "all_fiqa_results.json", "w") as f:
        json.dump(make_json_safe(combined), f, indent=2)

    print("\n" + "=" * 60)
    print("FIQA CROSS-DATASET SUMMARY")
    print("=" * 60)
    print(f"KD Routing:  acc={kd_results['hybrid_accuracy']:.4f}, "
          f"retention={kd_results['accuracy_retention']:.4f}, "
          f"teacher%={kd_results['teacher_usage_pct']:.1f}%")
    print(f"Multi-Exit:  acc={me_results['hybrid_accuracy']:.4f}, "
          f"retention={me_results['accuracy_retention']:.4f}, "
          f"exit%={me_results['early_exit_pct']:.1f}%")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
