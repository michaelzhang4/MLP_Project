"""Shared utilities for hybrid sentiment analysis experiments."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("FinancialPhraseBank-v1.0")

LABEL_LIST = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# Data Loading with Agreement Metadata
# ============================================================


def load_single_phrasebank(filepath: str, encoding: str = "latin-1") -> Dict[str, str]:
    """Load a single PhraseBank file. Returns {sentence: label}."""
    data = {}
    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if "@" in line:
                parts = line.rsplit("@", 1)
                if len(parts) == 2:
                    text, label = parts[0].strip(), parts[1].strip()
                    data[text] = label
    return data


def load_all_phrasebank(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load all 4 agreement levels and derive ambiguity metadata.
    Ambiguity score: 0.0 (in AllAgree = easy) to 1.0 (only in 50Agree = hard).
    """
    files = {
        "50": os.path.join(data_dir, "Sentences_50Agree.txt"),
        "66": os.path.join(data_dir, "Sentences_66Agree.txt"),
        "75": os.path.join(data_dir, "Sentences_75Agree.txt"),
        "100": os.path.join(data_dir, "Sentences_AllAgree.txt"),
    }

    datasets = {}
    for level, fp in files.items():
        if os.path.exists(fp):
            datasets[level] = load_single_phrasebank(fp)
        else:
            print(f"Warning: {fp} not found")

    all_sentences = datasets.get("50", {})
    if not all_sentences:
        raise FileNotFoundError("Sentences_50Agree.txt not found")

    rows = []
    for text, label in all_sentences.items():
        in_100 = text in datasets.get("100", {})
        in_75 = text in datasets.get("75", {})
        in_66 = text in datasets.get("66", {})

        if in_100:
            tier = "100"
            ambiguity = 0.0
        elif in_75:
            tier = "75"
            ambiguity = 0.33
        elif in_66:
            tier = "66"
            ambiguity = 0.67
        else:
            tier = "50"
            ambiguity = 1.0

        rows.append(
            {
                "text": text,
                "label": label,
                "agreement_tier": tier,
                "ambiguity_score": ambiguity,
            }
        )

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} sentences from FinancialPhraseBank")
    print(f"Agreement tiers: {df['agreement_tier'].value_counts().to_dict()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.2, seed=SEED):
    """Stratified train/val/test split."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=seed, stratify=train_df["label"]
    )
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# Statistical Testing & JSON Helpers
# ============================================================


def bootstrap_hypothesis_ci(
    labels: np.ndarray,
    hybrid_preds: np.ndarray,
    teacher_preds: np.ndarray,
    use_student: np.ndarray,
    n_bootstrap: int = 1000,
) -> Dict[str, List[float]]:
    n = len(labels)
    if n == 0:
        return {"retention_ci_95": [0.0, 0.0], "teacher_usage_ci_95": [0.0, 0.0]}
    rng = np.random.default_rng(42)
    ret = []
    usage = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        h = hybrid_preds[idx]
        t = teacher_preds[idx]
        u = use_student[idx]
        t_acc = accuracy_score(y, t)
        h_acc = accuracy_score(y, h)
        ret.append(h_acc / t_acc if t_acc > 0 else 0.0)
        usage.append((~u).mean() * 100.0)
    return {
        "retention_ci_95": [float(x) for x in np.percentile(ret, [2.5, 97.5])],
        "teacher_usage_ci_95": [float(x) for x in np.percentile(usage, [2.5, 97.5])],
    }


def mcnemar_test(hybrid_correct: np.ndarray, teacher_correct: np.ndarray) -> Dict[str, float]:
    b = np.sum(hybrid_correct & ~teacher_correct)
    c = np.sum(~hybrid_correct & teacher_correct)
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1.0 - stats.chi2.cdf(stat, df=1)
    return {"statistic": float(stat), "p_value": float(p)}


def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    return obj


# ============================================================
# Expected Calibration Error (ECE)
# ============================================================


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, Dict]:
    """Compute Expected Calibration Error.

    Args:
        confidences: (N,) predicted confidence (max softmax probability)
        accuracies: (N,) binary array, 1 if prediction was correct
        n_bins: number of equal-width bins

    Returns:
        ece: scalar ECE value
        bin_data: dict with per-bin statistics for reliability diagrams
    """
    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(float(accuracies[mask].mean()))
            bin_confs.append(float(confidences[mask].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2.0)

    n_total = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / n_total) * abs(bin_accs[i] - bin_confs[i])

    bin_data = {
        "bin_boundaries": bin_boundaries.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }
    return float(ece), bin_data


# ============================================================
# FiQA Dataset Loading
# ============================================================


def load_fiqa(
    sentiment_thresholds: Tuple[float, float] = (-0.1, 0.1),
) -> pd.DataFrame:
    """Load FiQA-SA dataset and discretize continuous scores to 3 classes.

    FiQA sentiment scores range from -1 to 1. We discretize using thresholds:
    - negative: score < low_threshold
    - neutral: low_threshold <= score <= high_threshold
    - positive: score > high_threshold

    Returns DataFrame with columns: text, label (matching PhraseBank schema).
    FiQA has no agreement tiers, so agreement_tier='fiqa' and ambiguity_score=0.5.
    """
    from datasets import load_dataset

    # Load all splits and combine
    ds_all = load_dataset("pauri32/fiqa-2018")
    items = []
    for split in ds_all:
        items.extend(ds_all[split])

    low, high = sentiment_thresholds
    rows = []
    for item in items:
        score = float(item["sentiment_score"])
        text = str(item["sentence"]).strip()
        if not text:
            continue

        if score < low:
            label = "negative"
        elif score > high:
            label = "positive"
        else:
            label = "neutral"

        rows.append({
            "text": text,
            "label": label,
            "agreement_tier": "fiqa",
            "ambiguity_score": 0.5,
            "sentiment_score": score,
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} sentences from FiQA-SA")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df
