"""
P3: Compute ECE (Expected Calibration Error) and generate reliability diagrams
for all models across experiments.

Runs inference on the test set for each model, collects confidences and
correctness, then computes ECE before/after temperature scaling.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import (
    DEVICE,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    expected_calibration_error,
    load_all_phrasebank,
    make_json_safe,
    set_seed,
    split_data,
)


# ============================================================
# Temperature Scaling (standalone, no dependency on hybrid_sentiment.py)
# ============================================================

class TemperatureScaler:
    """Post-hoc temperature scaling for calibration."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 500):
        """Fit temperature on validation logits."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        temp = torch.nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / torch.clamp(temp, min=0.01, max=50.0)
            loss = F.cross_entropy(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.clamp(temp, min=0.01, max=50.0).item())

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


# ============================================================
# Model inference helpers
# ============================================================

def predict_with_logits(
    model, tokenizer, texts: List[str], batch_size: int = 32, device=DEVICE,
    reorder_idx=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference, return (pred_labels, confidences, logits)."""
    model.eval()
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        if reorder_idx is not None:
            logits = logits[:, reorder_idx]
        all_logits.append(logits)

    logits = np.vstack(all_logits)
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    confidences = np.max(probs, axis=1)
    pred_ids = np.argmax(logits, axis=1)
    pred_labels = np.array([LABEL_LIST[i] for i in pred_ids])
    return pred_labels, confidences, logits


def get_reorder_idx(model) -> list:
    """Get index mapping from model's label order to LABEL_LIST order."""
    model_labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label_to_model_idx = {v: k for k, v in model_labels.items()}
    return [label_to_model_idx[lbl] for lbl in LABEL_LIST]


# ============================================================
# Reliability diagram plotting
# ============================================================

def plot_reliability_diagram(
    model_results: Dict[str, Dict],
    out_path: str,
    title: str = "Reliability Diagrams",
):
    """Plot reliability diagrams for multiple models."""
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)

    for idx, (name, data) in enumerate(model_results.items()):
        ax = axes[0, idx]
        bins = data["bin_data"]
        bin_accs = np.array(bins["bin_accuracies"])
        bin_confs = np.array(bins["bin_confidences"])
        bin_counts = np.array(bins["bin_counts"])
        boundaries = np.array(bins["bin_boundaries"])
        bin_centers = (boundaries[:-1] + boundaries[1:]) / 2
        widths = boundaries[1:] - boundaries[:-1]

        # Bar chart of accuracy per bin
        mask = np.array(bin_counts) > 0
        ax.bar(
            bin_centers[mask], bin_accs[mask], width=widths[mask] * 0.9,
            alpha=0.7, color="steelblue", edgecolor="navy", label="Accuracy"
        )
        # Gap bars (overconfidence)
        gap = bin_confs[mask] - bin_accs[mask]
        ax.bar(
            bin_centers[mask], gap, bottom=bin_accs[mask],
            width=widths[mask] * 0.9, alpha=0.3, color="coral",
            edgecolor="darkred", label="Gap"
        )
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ece = data["ece"]
        acc = data.get("accuracy", 0)
        ax.set_title(f"{name}\nECE={ece:.4f}, Acc={acc:.4f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved reliability diagram to {out_path}")


def plot_ece_comparison_bar(
    ece_before: Dict[str, float],
    ece_after: Dict[str, float],
    out_path: str,
):
    """Bar chart comparing ECE before/after temperature scaling."""
    models = list(ece_before.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width / 2, [ece_before[m] for m in models], width,
                   label="Before T-scaling", color="coral", alpha=0.8)
    bars2 = ax.bar(x + width / 2, [ece_after[m] for m in models], width,
                   label="After T-scaling", color="steelblue", alpha=0.8)

    ax.set_ylabel("ECE")
    ax.set_title("Expected Calibration Error: Before vs After Temperature Scaling")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved ECE comparison to {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compute ECE and reliability diagrams")
    parser.add_argument("--data-dir", type=str, default="FinancialPhraseBank-v1.0")
    parser.add_argument("--results-dir", type=str, default="results_ece")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(exist_ok=True)
    set_seed(args.seed)

    print(f"Device: {DEVICE}")

    # Load data
    print("\n=== Loading data ===")
    df = load_all_phrasebank(args.data_dir)
    train_df, val_df, test_df = split_data(df)
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels])
    test_texts = test_df["text"].tolist()
    test_labels = np.array(test_df["label"].tolist())

    # Models to evaluate
    model_configs = {
        "FinBERT": "ProsusAI/finbert",
        "DistilRoBERTa-Financial": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    }

    all_results = {}
    ece_before = {}
    ece_after = {}
    model_results_before = {}
    model_results_after = {}

    for name, model_id in model_configs.items():
        print(f"\n=== {name} ({model_id}) ===")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.to(DEVICE)
        except Exception as e:
            print(f"  Failed to load {model_id}: {e}")
            continue

        # Determine label reordering
        try:
            reorder_idx = get_reorder_idx(model)
        except (KeyError, AttributeError):
            reorder_idx = None

        # Validation inference (for temperature fitting)
        val_preds, val_confs, val_logits = predict_with_logits(
            model, tokenizer, val_texts, batch_size=args.batch_size, reorder_idx=reorder_idx
        )

        # Test inference
        test_preds, test_confs, test_logits = predict_with_logits(
            model, tokenizer, test_texts, batch_size=args.batch_size, reorder_idx=reorder_idx
        )
        correct = (test_preds == test_labels).astype(float)
        acc = float(correct.mean())
        f1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))

        # ECE before temperature scaling
        ece_raw, bin_data_raw = expected_calibration_error(test_confs, correct)
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(f"  ECE (before T-scaling): {ece_raw:.4f}")

        ece_before[name] = ece_raw
        model_results_before[name] = {"ece": ece_raw, "bin_data": bin_data_raw, "accuracy": acc}

        # Temperature scaling
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_label_ids)
        cal_probs = scaler.calibrate(test_logits)
        cal_confs = np.max(cal_probs, axis=1)
        cal_preds = np.array([LABEL_LIST[i] for i in np.argmax(cal_probs, axis=1)])
        cal_correct = (cal_preds == test_labels).astype(float)

        ece_cal, bin_data_cal = expected_calibration_error(cal_confs, cal_correct)
        print(f"  Temperature: {scaler.temperature:.4f}")
        print(f"  ECE (after T-scaling): {ece_cal:.4f}")

        ece_after[name] = ece_cal
        model_results_after[name] = {
            "ece": ece_cal, "bin_data": bin_data_cal,
            "accuracy": float(cal_correct.mean()),
        }

        all_results[name] = {
            "model_id": model_id,
            "accuracy": acc,
            "f1_macro": f1,
            "ece_before_tscaling": ece_raw,
            "ece_after_tscaling": ece_cal,
            "temperature": scaler.temperature,
            "bin_data_before": bin_data_raw,
            "bin_data_after": bin_data_cal,
        }

        del model
        torch.cuda.empty_cache()

    # Generate plots
    print("\n=== Generating plots ===")
    if model_results_before:
        plot_reliability_diagram(
            model_results_before,
            str(out_dir / "reliability_before_tscaling.png"),
            "Reliability Diagrams (Before Temperature Scaling)",
        )
        plot_reliability_diagram(
            model_results_after,
            str(out_dir / "reliability_after_tscaling.png"),
            "Reliability Diagrams (After Temperature Scaling)",
        )

    if ece_before and ece_after:
        plot_ece_comparison_bar(ece_before, ece_after, str(out_dir / "ece_comparison.png"))

    # Save results
    out_json = out_dir / "ece_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(all_results), f, indent=2)
    print(f"\nSaved ECE results to {out_json}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<35} {'Acc':>6} {'F1':>6} {'ECE-raw':>8} {'ECE-cal':>8} {'Temp':>6}")
    print("-" * 70)
    for name, res in all_results.items():
        print(f"{name:<35} {res['accuracy']:>6.4f} {res['f1_macro']:>6.4f} "
              f"{res['ece_before_tscaling']:>8.4f} {res['ece_after_tscaling']:>8.4f} "
              f"{res['temperature']:>6.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
