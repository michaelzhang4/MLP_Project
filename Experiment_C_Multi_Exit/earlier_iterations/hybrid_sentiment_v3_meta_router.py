"""
Variant 3: Meta-learner router.

Student:
- DistilBERT (CE fine-tuning, optional KD)

Teacher:
- FinBERT

Router:
- RandomForest predicts probability that student is wrong.
- Route by score: use student if (1 - p_wrong) >= tau, else defer to teacher.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from hybrid_sentiment import (
    CostAwareOptimizer,
    FinBERTAnalyzer,
    TemperatureScaler,
)
from hybrid_sentiment_v2 import (
    DistilStudent,
    TrainConfig,
    reorder_teacher_logits,
    route_predictions,
)
from utils import (
    LABEL2ID,
    bootstrap_hypothesis_ci,
    load_all_phrasebank,
    make_json_safe,
    mcnemar_test,
    set_seed,
    split_data,
)


ADVERSARIAL_TOKENS = (
    "but",
    "however",
    "despite",
    "although",
    "though",
    "yet",
    "nevertheless",
    "nonetheless",
)


def _build_router_features(
    probs: np.ndarray, texts: List[str]
) -> Tuple[np.ndarray, List[str]]:
    conf = np.max(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    lengths = np.array([len(t.split()) for t in texts], dtype=float)
    has_adv = np.array(
        [
            float(any(tok in t.lower().split() for tok in ADVERSARIAL_TOKENS))
            for t in texts
        ],
        dtype=float,
    )
    adv_count = np.array(
        [sum(t.lower().count(tok) for tok in ADVERSARIAL_TOKENS) for t in texts],
        dtype=float,
    )

    feats = np.column_stack(
        [
            probs[:, 0],
            probs[:, 1],
            probs[:, 2],
            conf,
            margin,
            entropy,
            lengths,
            has_adv,
            adv_count,
        ]
    )
    names = [
        "p_negative",
        "p_neutral",
        "p_positive",
        "max_conf",
        "top_margin",
        "entropy",
        "length_words",
        "has_adversarial_token",
        "adversarial_token_count",
    ]
    return feats, names


def main():
    parser = argparse.ArgumentParser(description="Hybrid v3 with meta-learner router")
    parser.add_argument("--data-dir", type=str, default="FinancialPhraseBank-v1.0")
    parser.add_argument("--results-dir", type=str, default="results_v3_meta_router")
    parser.add_argument("--student-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--student-max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ce-epochs", type=int, default=4)
    parser.add_argument("--kd-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--router-trees", type=int, default=400)
    parser.add_argument("--router-max-depth", type=int, default=8)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--target-retention", type=float, default=0.98)
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
    student = DistilStudent(
        model_name=args.student_model, max_length=args.student_max_length
    )
    teacher = FinBERTAnalyzer()

    cfg = TrainConfig(
        ce_epochs=args.ce_epochs,
        kd_epochs=args.kd_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.student_max_length,
        alpha=args.alpha,
        kd_temperature=args.kd_temperature,
    )

    print("=== Train student CE ===")
    ce_history = student.train_ce(train_texts, train_labels, val_texts, val_labels, cfg)

    kd_history = []
    if args.use_kd:
        print("=== Train student KD ===")
        train_teacher_logits, _ = teacher.get_logits_batch(train_texts, batch_size=32)
        val_teacher_logits, _ = teacher.get_logits_batch(val_texts, batch_size=32)
        train_teacher_logits = reorder_teacher_logits(train_teacher_logits, teacher.label_map)
        val_teacher_logits = reorder_teacher_logits(val_teacher_logits, teacher.label_map)
        kd_history = student.train_kd(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            train_teacher_logits=train_teacher_logits,
            val_teacher_logits=val_teacher_logits,
            cfg=cfg,
        )

    print("=== Student outputs ===")
    train_student_preds, _, train_student_logits = student.predict_batch(
        train_texts, batch_size=32
    )
    val_student_preds, _, val_student_logits = student.predict_batch(val_texts, batch_size=32)
    test_student_preds, _, test_student_logits = student.predict_batch(
        test_texts, batch_size=32
    )

    temp_scaler = TemperatureScaler()
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels], dtype=int)
    temp_scaler.fit(val_student_logits, val_label_ids)

    train_probs_cal = temp_scaler.calibrate(train_student_logits)
    val_probs_cal = temp_scaler.calibrate(val_student_logits)
    test_probs_cal = temp_scaler.calibrate(test_student_logits)

    print("=== Teacher outputs ===")
    val_teacher_preds = np.array(
        [r[0] for r in teacher.analyze_batch(val_texts, batch_size=32)], dtype=object
    )
    test_teacher_preds = np.array(
        [r[0] for r in teacher.analyze_batch(test_texts, batch_size=32)], dtype=object
    )

    print("=== Train meta-router ===")
    train_features, feature_names = _build_router_features(train_probs_cal, train_texts)
    val_features, _ = _build_router_features(val_probs_cal, val_texts)
    test_features, _ = _build_router_features(test_probs_cal, test_texts)

    train_labels_arr = np.array(train_labels, dtype=object)
    train_wrong = (train_student_preds != train_labels_arr).astype(int)

    router = RandomForestClassifier(
        n_estimators=args.router_trees,
        max_depth=args.router_max_depth,
        min_samples_leaf=3,
        random_state=args.seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    router.fit(train_features, train_wrong)

    val_prob_wrong = router.predict_proba(val_features)[:, 1]
    val_routing_scores = 1.0 - val_prob_wrong
    val_labels_arr = np.array(val_labels, dtype=object)
    val_teacher_acc = accuracy_score(val_labels_arr, val_teacher_preds)

    optimizer = CostAwareOptimizer()
    opt_results = optimizer.optimize(
        val_routing_scores,
        val_student_preds,
        val_teacher_preds,
        val_labels_arr,
    )
    target = optimizer.find_target_operating_point(
        val_routing_scores,
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

    print("=== Test evaluation ===")
    test_prob_wrong = router.predict_proba(test_features)[:, 1]
    test_routing_scores = 1.0 - test_prob_wrong
    hybrid_preds, use_student = route_predictions(
        student_preds=test_student_preds,
        teacher_preds=test_teacher_preds,
        student_confidence=test_routing_scores,
        tau=tau,
    )

    teacher_acc = accuracy_score(test_labels, test_teacher_preds)
    teacher_f1 = f1_score(test_labels, test_teacher_preds, average="macro", zero_division=0)
    hybrid_acc = accuracy_score(test_labels, hybrid_preds)
    hybrid_f1 = f1_score(test_labels, hybrid_preds, average="macro", zero_division=0)
    retention = hybrid_acc / teacher_acc if teacher_acc > 0 else 0.0
    teacher_pct = float((~use_student).mean() * 100.0)

    ci = bootstrap_hypothesis_ci(
        labels=np.array(test_labels, dtype=object),
        hybrid_preds=hybrid_preds,
        teacher_preds=test_teacher_preds,
        use_student=use_student,
        n_bootstrap=args.n_bootstrap,
    )
    mcn = mcnemar_test(
        hybrid_correct=(hybrid_preds == np.array(test_labels, dtype=object)),
        teacher_correct=(test_teacher_preds == np.array(test_labels, dtype=object)),
    )

    print("=== Efficiency ===")
    t0 = time.time()
    rt_student_preds, _, rt_student_logits = student.predict_batch(test_texts, batch_size=32)
    student_time = time.time() - t0

    t0 = time.time()
    rt_probs = temp_scaler.calibrate(rt_student_logits)
    rt_feats, _ = _build_router_features(rt_probs, test_texts)
    rt_prob_wrong = router.predict_proba(rt_feats)[:, 1]
    rt_scores = 1.0 - rt_prob_wrong
    route_time = time.time() - t0

    deferred_idx = np.where(rt_scores < tau)[0]
    deferred_texts = [test_texts[i] for i in deferred_idx.tolist()]
    t0 = time.time()
    deferred_teacher_preds = np.array([], dtype=object)
    if len(deferred_texts) > 0:
        deferred_teacher_preds = np.array(
            [r[0] for r in teacher.analyze_batch(deferred_texts, batch_size=32)],
            dtype=object,
        )
    teacher_deferred_time = time.time() - t0

    rt_hybrid = rt_student_preds.copy()
    if len(deferred_idx) > 0:
        rt_hybrid[deferred_idx] = deferred_teacher_preds
    hybrid_total_time = student_time + route_time + teacher_deferred_time

    n = len(test_texts)
    teacher_total_time = float(
        time.time() - time.time()
    )  # placeholder to keep structure explicit
    t0 = time.time()
    _ = teacher.analyze_batch(test_texts, batch_size=32)
    teacher_total_time = time.time() - t0

    feature_importance = {
        name: float(score)
        for name, score in zip(feature_names, router.feature_importances_)
    }

    results = {
        "version": "v3_meta_router",
        "config": vars(args),
        "training": {
            "ce_history": ce_history,
            "kd_enabled": bool(args.use_kd),
            "kd_history": kd_history,
        },
        "router": {
            "type": "RandomForestClassifier",
            "feature_importance": feature_importance,
        },
        "threshold_optimization": {
            "validation_teacher_acc": float(val_teacher_acc),
            "opt_results": {str(k): v for k, v in opt_results.items()},
            "target_operating_point": target,
            "selected_tau": float(tau),
        },
        "exp_hypothesis": {
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
        "efficiency": {
            "FinBERT": {
                "total_time": float(teacher_total_time),
                "throughput": float(n / max(teacher_total_time, 1e-9)),
            },
            "DistilBERT": {
                "total_time": float(student_time),
                "throughput": float(n / max(student_time, 1e-9)),
            },
            "Hybrid-v3-meta": {
                "total_time": float(hybrid_total_time),
                "throughput": float(n / max(hybrid_total_time, 1e-9)),
                "student_stage_time": float(student_time),
                "router_stage_time": float(route_time),
                "teacher_stage_time": float(teacher_deferred_time),
                "teacher_usage_pct": float((len(deferred_idx) / max(1, n)) * 100.0),
            },
        },
    }

    out_json = out_dir / "all_results_v3_meta_router.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
