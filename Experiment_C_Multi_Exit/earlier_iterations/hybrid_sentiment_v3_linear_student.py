"""
Variant 1: Linear-on-Embeddings student.

Student:
- TF-IDF sentence embedding
- Logistic Regression sentiment classifier

Teacher:
- FinBERT

Router:
- Calibrated student confidence threshold
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow imports from the project root and parent experiment folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from hybrid_sentiment import (
    CostAwareOptimizer,
    FinBERTAnalyzer,
    TemperatureScaler,
    VADERAnalyzer,
)
from hybrid_sentiment_v2 import route_predictions
from utils import (
    LABEL2ID,
    LABEL_LIST,
    bootstrap_hypothesis_ci,
    load_all_phrasebank,
    make_json_safe,
    mcnemar_test,
    set_seed,
    split_data,
)


class LinearEmbeddingStudent:
    """Fast linear student over sparse lexical embeddings."""

    def __init__(
        self,
        max_features: int = 50000,
        ngram_min: int = 1,
        ngram_max: int = 2,
        min_df: int = 2,
        c: float = 6.0,
    ):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(ngram_min, ngram_max),
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            max_iter=2000,
            C=c,
            solver="lbfgs",
        )

    def fit(self, texts: List[str], labels: List[str]):
        x = self.vectorizer.fit_transform(texts)
        self.model.fit(x, labels)

    def predict_batch(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self.vectorizer.transform(texts)
        probs_raw = self.model.predict_proba(x)
        logits_raw = self.model.decision_function(x)
        if logits_raw.ndim == 1:
            logits_raw = np.stack([-logits_raw, logits_raw], axis=1)

        classes = list(self.model.classes_)
        cls_idx = [classes.index(lbl) for lbl in LABEL_LIST]
        probs = probs_raw[:, cls_idx]
        logits = logits_raw[:, cls_idx]
        conf = np.max(probs, axis=1)
        preds = np.array([LABEL_LIST[i] for i in np.argmax(probs, axis=1)], dtype=object)
        return preds, conf, logits, probs


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid v3 linear student (TF-IDF + Logistic Regression)"
    )
    parser.add_argument("--data-dir", type=str, default="FinancialPhraseBank-v1.0")
    parser.add_argument("--results-dir", type=str, default="results_v3_linear_student")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c", type=float, default=6.0)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--target-retention", type=float, default=0.95)
    parser.add_argument("--max-teacher-pct", type=float, default=30.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(exist_ok=True)
    set_seed(args.seed)

    print("=== Data ===")
    df = load_all_phrasebank(args.data_dir)
    train_df, val_df, test_df = split_data(
        df, test_size=args.test_split, val_size=args.val_split
    )
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print("=== Models ===")
    student = LinearEmbeddingStudent(
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        c=args.c,
    )
    student.fit(train_texts, train_labels)
    teacher = FinBERTAnalyzer()
    vader = VADERAnalyzer()

    print("=== Validation calibration and threshold selection ===")
    val_student_preds, _, val_student_logits, _ = student.predict_batch(val_texts)
    val_teacher = teacher.analyze_batch(val_texts, batch_size=32)
    val_teacher_preds = np.array([r[0] for r in val_teacher], dtype=object)
    val_labels_arr = np.array(val_labels, dtype=object)

    temp_scaler = TemperatureScaler()
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels], dtype=int)
    temp_scaler.fit(val_student_logits, val_label_ids)
    val_cal_probs = temp_scaler.calibrate(val_student_logits)
    val_cal_conf = np.max(val_cal_probs, axis=1)

    optimizer = CostAwareOptimizer()
    val_teacher_acc = accuracy_score(val_labels_arr, val_teacher_preds)
    opt_results = optimizer.optimize(
        val_cal_conf,
        val_student_preds,
        val_teacher_preds,
        val_labels_arr,
    )
    target = optimizer.find_target_operating_point(
        val_cal_conf,
        val_student_preds,
        val_teacher_preds,
        val_labels_arr,
        finbert_acc=val_teacher_acc,
        target_retention=args.target_retention,
        max_finbert_pct=args.max_teacher_pct,
    )
    if target and target["meets_target"]:
        tau = float(target["tau"])
    else:
        tau = float(min(opt_results.items(), key=lambda kv: kv[1]["cost"])[1]["tau"])
    print(f"Selected tau={tau:.3f}")

    print("=== Experiment A: Baselines ===")
    n_test = len(test_texts)

    t0 = time.time()
    vader_results = vader.analyze_batch(test_texts)
    vader_time = time.time() - t0
    vader_preds = np.array([r[0] for r in vader_results], dtype=object)

    t0 = time.time()
    student_preds, _, student_logits, _ = student.predict_batch(test_texts)
    student_time = time.time() - t0
    student_probs_cal = temp_scaler.calibrate(student_logits)
    student_conf_cal = np.max(student_probs_cal, axis=1)

    t0 = time.time()
    teacher_results = teacher.analyze_batch(test_texts, batch_size=32)
    teacher_time = time.time() - t0
    teacher_preds = np.array([r[0] for r in teacher_results], dtype=object)

    exp_a = {
        "VADER": {
            "accuracy": float(accuracy_score(test_labels, vader_preds)),
            "f1_macro": float(
                f1_score(test_labels, vader_preds, average="macro", zero_division=0)
            ),
            "avg_time_ms": float((vader_time / n_test) * 1000.0),
        },
        "LinearStudent": {
            "accuracy": float(accuracy_score(test_labels, student_preds)),
            "f1_macro": float(
                f1_score(test_labels, student_preds, average="macro", zero_division=0)
            ),
            "avg_time_ms": float((student_time / n_test) * 1000.0),
        },
        "FinBERT": {
            "accuracy": float(accuracy_score(test_labels, teacher_preds)),
            "f1_macro": float(
                f1_score(test_labels, teacher_preds, average="macro", zero_division=0)
            ),
            "avg_time_ms": float((teacher_time / n_test) * 1000.0),
        },
    }

    print("=== Experiment B: Cascaded hypothesis ===")
    hybrid_preds, use_student = route_predictions(
        student_preds=student_preds,
        teacher_preds=teacher_preds,
        student_confidence=student_conf_cal,
        tau=tau,
    )
    teacher_acc = accuracy_score(test_labels, teacher_preds)
    hybrid_acc = accuracy_score(test_labels, hybrid_preds)
    teacher_f1 = f1_score(test_labels, teacher_preds, average="macro", zero_division=0)
    hybrid_f1 = f1_score(test_labels, hybrid_preds, average="macro", zero_division=0)
    retention = hybrid_acc / teacher_acc if teacher_acc > 0 else 0.0
    teacher_pct = float((~use_student).mean() * 100.0)

    ci = bootstrap_hypothesis_ci(
        labels=np.array(test_labels, dtype=object),
        hybrid_preds=hybrid_preds,
        teacher_preds=teacher_preds,
        use_student=use_student,
        n_bootstrap=args.n_bootstrap,
    )
    mcn = mcnemar_test(
        hybrid_correct=(hybrid_preds == np.array(test_labels, dtype=object)),
        teacher_correct=(teacher_preds == np.array(test_labels, dtype=object)),
    )

    print("=== Experiment C: Efficiency ===")
    t0 = time.time()
    rt_student_preds, _, rt_student_logits, _ = student.predict_batch(test_texts)
    rt_student_probs = temp_scaler.calibrate(rt_student_logits)
    rt_conf = np.max(rt_student_probs, axis=1)
    student_stage_time = time.time() - t0

    deferred_idx = np.where(rt_conf < tau)[0]
    deferred_texts = [test_texts[i] for i in deferred_idx.tolist()]

    t0 = time.time()
    deferred_teacher_preds = np.array([], dtype=object)
    if len(deferred_texts) > 0:
        deferred_teacher_res = teacher.analyze_batch(deferred_texts, batch_size=32)
        deferred_teacher_preds = np.array([r[0] for r in deferred_teacher_res], dtype=object)
    teacher_stage_time = time.time() - t0

    rt_hybrid = rt_student_preds.copy()
    if len(deferred_idx) > 0:
        rt_hybrid[deferred_idx] = deferred_teacher_preds

    hybrid_total_time = student_stage_time + teacher_stage_time
    exp_c = {
        "FinBERT": {
            "total_time": float(teacher_time),
            "throughput": float(n_test / max(teacher_time, 1e-9)),
        },
        "LinearStudent": {
            "total_time": float(student_time),
            "throughput": float(n_test / max(student_time, 1e-9)),
        },
        "Hybrid-v3-linear": {
            "total_time": float(hybrid_total_time),
            "throughput": float(n_test / max(hybrid_total_time, 1e-9)),
            "teacher_usage_pct": float((len(deferred_idx) / max(1, n_test)) * 100.0),
        },
    }

    results = {
        "version": "v3_linear_student",
        "config": vars(args),
        "threshold_optimization": {
            "validation_teacher_acc": float(val_teacher_acc),
            "opt_results": {str(k): v for k, v in opt_results.items()},
            "target_operating_point": target,
            "selected_tau": float(tau),
        },
        "expA_baselines": exp_a,
        "expB_hypothesis": {
            "teacher_accuracy": float(teacher_acc),
            "teacher_f1": float(teacher_f1),
            "hybrid_accuracy": float(hybrid_acc),
            "hybrid_f1": float(hybrid_f1),
            "accuracy_retention": float(retention),
            "teacher_usage_pct": float(teacher_pct),
            "selected_tau": float(tau),
            "target_retention": float(args.target_retention),
            "target_max_teacher_pct": float(args.max_teacher_pct),
            "hypothesis_accuracy_met": bool(retention >= args.target_retention),
            "hypothesis_cost_met": bool(teacher_pct <= args.max_teacher_pct),
            "ci_retention_95": ci["retention_ci_95"],
            "ci_teacher_usage_95": ci["teacher_usage_ci_95"],
            "mcnemar_statistic": mcn["statistic"],
            "mcnemar_p_value": mcn["p_value"],
        },
        "expC_efficiency": exp_c,
    }

    out_json = out_dir / "all_results_v3_linear_student.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
