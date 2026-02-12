"""Shared utilities for hybrid sentiment analysis experiments."""

import os
import random
from pathlib import Path
from typing import Dict, List

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
