"""
P4: Evaluate off-the-shelf HuggingFace financial sentiment models.

Preempts 'why not just use X?' reviewer objection.
Note: many of these models are trained on overlapping PhraseBank data,
so high accuracy may reflect data leakage rather than generalization.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils import (
    DEVICE,
    LABEL_LIST,
    LABEL2ID,
    expected_calibration_error,
    load_all_phrasebank,
    make_json_safe,
    set_seed,
    split_data,
)


# ============================================================
# HuggingFace model registry
# ============================================================

HF_MODELS = {
    "FinBERT (ProsusAI)": {
        "model_id": "ProsusAI/finbert",
        "label_map": None,  # auto-detect from config
    },
    "DistilRoBERTa-Financial": {
        "model_id": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "label_map": None,
    },
    "FinancialBERT-Sentiment": {
        "model_id": "ahmedrachid/FinancialBERT-Sentiment-Analysis",
        "label_map": None,
    },
}


# ============================================================
# Inference
# ============================================================

def get_label_reorder(model) -> list:
    """Determine reordering from model's label space to LABEL_LIST."""
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label_to_idx = {v: k for k, v in id2label.items()}
    try:
        return [label_to_idx[lbl] for lbl in LABEL_LIST]
    except KeyError:
        return None


def evaluate_model(
    model_id: str,
    test_texts: List[str],
    test_labels: np.ndarray,
    batch_size: int = 32,
) -> Dict:
    """Load and evaluate a HuggingFace model."""
    print(f"\n  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(DEVICE)
    model.eval()

    reorder_idx = get_label_reorder(model)
    id2label_model = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

    all_logits = []
    t0 = time.time()
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        if reorder_idx is not None:
            logits = logits[:, reorder_idx]
        all_logits.append(logits)
    elapsed = time.time() - t0

    logits = np.vstack(all_logits)
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    confidences = np.max(probs, axis=1)

    if reorder_idx is not None:
        pred_ids = np.argmax(logits, axis=1)
        preds = np.array([LABEL_LIST[i] for i in pred_ids])
    else:
        # Fallback: use model's own label mapping
        pred_ids = np.argmax(logits, axis=1)
        preds = np.array([id2label_model.get(i, "neutral") for i in pred_ids])

    correct = (preds == test_labels).astype(float)
    acc = float(accuracy_score(test_labels, preds))
    f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))
    ece, bin_data = expected_calibration_error(confidences, correct)
    cm = confusion_matrix(test_labels, preds, labels=LABEL_LIST).tolist()
    report = classification_report(test_labels, preds, output_dict=True, zero_division=0)

    n = len(test_texts)
    throughput = n / max(elapsed, 1e-9)

    del model
    torch.cuda.empty_cache()

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "ece": ece,
        "bin_data": bin_data,
        "confusion_matrix": cm,
        "classification_report": report,
        "inference_time_s": float(elapsed),
        "throughput_samples_per_s": float(throughput),
        "n_samples": n,
        "logits_shape": list(logits.shape),
    }


# ============================================================
# Plotting
# ============================================================

def plot_comparison(results: Dict[str, Dict], out_dir: Path):
    """Bar chart comparing all HF models."""
    models = list(results.keys())
    accs = [results[m]["accuracy"] for m in models]
    f1s = [results[m]["f1_macro"] for m in models]
    eces = [results[m]["ece"] for m in models]
    throughputs = [results[m]["throughput_samples_per_s"] for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Accuracy
    bars = axes[0, 0].bar(range(len(models)), accs, color="steelblue", alpha=0.8)
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Test Accuracy")
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    axes[0, 0].set_ylim(0, 1.0)
    for bar, val in zip(bars, accs):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # F1
    bars = axes[0, 1].bar(range(len(models)), f1s, color="seagreen", alpha=0.8)
    axes[0, 1].set_ylabel("F1 Macro")
    axes[0, 1].set_title("F1 Score (Macro)")
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    axes[0, 1].set_ylim(0, 1.0)
    for bar, val in zip(bars, f1s):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # ECE
    bars = axes[1, 0].bar(range(len(models)), eces, color="coral", alpha=0.8)
    axes[1, 0].set_ylabel("ECE")
    axes[1, 0].set_title("Expected Calibration Error")
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    for bar, val in zip(bars, eces):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Throughput
    bars = axes[1, 1].bar(range(len(models)), throughputs, color="mediumpurple", alpha=0.8)
    axes[1, 1].set_ylabel("Samples/sec")
    axes[1, 1].set_title("Inference Throughput")
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    for bar, val in zip(bars, throughputs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Off-the-Shelf HuggingFace Financial Sentiment Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "hf_baselines_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison plot to {out_dir / 'hf_baselines_comparison.png'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate off-the-shelf HF financial sentiment models")
    parser.add_argument("--data-dir", type=str, default="FinancialPhraseBank-v1.0")
    parser.add_argument("--results-dir", type=str, default="results_hf_baselines")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(exist_ok=True)
    set_seed(args.seed)

    print(f"Device: {DEVICE}")

    # Load data (same split as all experiments)
    print("\n=== Loading data ===")
    df = load_all_phrasebank(args.data_dir)
    _, _, test_df = split_data(df)
    test_texts = test_df["text"].tolist()
    test_labels = np.array(test_df["label"].tolist())

    print(f"Test set: {len(test_texts)} samples")
    print(f"Label distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    # Evaluate each model
    results = {}
    for name, config in HF_MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        try:
            res = evaluate_model(
                model_id=config["model_id"],
                test_texts=test_texts,
                test_labels=test_labels,
                batch_size=args.batch_size,
            )
            res["model_id"] = config["model_id"]
            results[name] = res
            print(f"  Accuracy: {res['accuracy']:.4f}")
            print(f"  F1 Macro: {res['f1_macro']:.4f}")
            print(f"  ECE:      {res['ece']:.4f}")
            print(f"  Time:     {res['inference_time_s']:.2f}s ({res['throughput_samples_per_s']:.0f} samples/s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No models evaluated successfully!")
        return

    # Plot comparison
    print("\n=== Generating plots ===")
    plot_comparison(results, out_dir)

    # Save results
    out_json = out_dir / "hf_baselines_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)
    print(f"\nSaved results to {out_json}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'Acc':>6} {'F1':>6} {'ECE':>8} {'Throughput':>12} {'Note'}")
    print("-" * 80)
    for name, res in results.items():
        note = ""
        if res["accuracy"] > 0.85:
            note = "(possible data leakage)"
        print(f"{name:<30} {res['accuracy']:>6.4f} {res['f1_macro']:>6.4f} "
              f"{res['ece']:>8.4f} {res['throughput_samples_per_s']:>9.0f}/s  {note}")
    print("=" * 80)
    print("\nNote: Models trained on PhraseBank data may show inflated accuracy due to data leakage.")
    print("Our routing framework is orthogonal and can accelerate any of these backbones.")


if __name__ == "__main__":
    main()
