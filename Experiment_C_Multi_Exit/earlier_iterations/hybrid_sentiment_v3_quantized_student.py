"""
Variant 4: Quantized student (INT8 dynamic quantization).

Student:
- DistilBERT trained in FP32 (CE + optional KD)
- Inference with dynamically quantized INT8 linear layers on CPU

Teacher:
- FinBERT

Router:
- Calibrated student confidence threshold
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

# Allow imports from the project root and parent experiment folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.ao.quantization import quantize_dynamic

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
    ID2LABEL,
    LABEL2ID,
    bootstrap_hypothesis_ci,
    load_all_phrasebank,
    make_json_safe,
    mcnemar_test,
    set_seed,
    split_data,
)


def predict_transformer_cpu(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    model.to(torch.device("cpu"))

    preds = []
    confs = []
    logits_out = []

    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        logits_out.append(logits.detach().cpu().numpy())
        confs.extend(conf.detach().cpu().numpy().tolist())
        preds.extend([ID2LABEL[int(p)] for p in pred.detach().cpu().numpy().tolist()])

    elapsed = time.time() - t0
    return np.array(preds, dtype=object), np.array(confs), np.vstack(logits_out), elapsed


def main():
    parser = argparse.ArgumentParser(description="Hybrid v3 quantized student")
    parser.add_argument("--data-dir", type=str, default="FinancialPhraseBank-v1.0")
    parser.add_argument("--results-dir", type=str, default="results_v3_quantized_student")
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

    print("=== Train student ===")
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
    ce_history = student.train_ce(train_texts, train_labels, val_texts, val_labels, cfg)

    kd_history = []
    if args.use_kd:
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

    print("=== Build FP32 CPU and INT8 student models ===")
    fp32_cpu_model = copy.deepcopy(student.model).cpu().eval()
    int8_model = quantize_dynamic(
        copy.deepcopy(student.model).cpu().eval(),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    print("=== Validation calibration and threshold selection (INT8) ===")
    _, _, val_logits_int8, _ = predict_transformer_cpu(
        int8_model,
        student.tokenizer,
        val_texts,
        max_length=args.student_max_length,
        batch_size=32,
    )
    val_student_preds_int8, _, _, _ = predict_transformer_cpu(
        int8_model,
        student.tokenizer,
        val_texts,
        max_length=args.student_max_length,
        batch_size=32,
    )
    val_teacher_preds = np.array(
        [r[0] for r in teacher.analyze_batch(val_texts, batch_size=32)], dtype=object
    )
    val_labels_arr = np.array(val_labels, dtype=object)

    temp_scaler = TemperatureScaler()
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels], dtype=int)
    temp_scaler.fit(val_logits_int8, val_label_ids)
    val_probs_cal = temp_scaler.calibrate(val_logits_int8)
    val_conf_cal = np.max(val_probs_cal, axis=1)

    optimizer = CostAwareOptimizer()
    val_teacher_acc = accuracy_score(val_labels_arr, val_teacher_preds)
    opt_results = optimizer.optimize(
        val_conf_cal,
        val_student_preds_int8,
        val_teacher_preds,
        val_labels_arr,
    )
    target = optimizer.find_target_operating_point(
        val_conf_cal,
        val_student_preds_int8,
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

    print("=== Test baselines ===")
    fp32_preds, _, _, fp32_time = predict_transformer_cpu(
        fp32_cpu_model,
        student.tokenizer,
        test_texts,
        max_length=args.student_max_length,
        batch_size=32,
    )
    int8_preds, _, int8_logits, int8_time = predict_transformer_cpu(
        int8_model,
        student.tokenizer,
        test_texts,
        max_length=args.student_max_length,
        batch_size=32,
    )
    teacher_t0 = time.time()
    teacher_preds = np.array(
        [r[0] for r in teacher.analyze_batch(test_texts, batch_size=32)], dtype=object
    )
    teacher_time = time.time() - teacher_t0

    int8_probs_cal = temp_scaler.calibrate(int8_logits)
    int8_conf_cal = np.max(int8_probs_cal, axis=1)
    hybrid_preds, use_student = route_predictions(
        student_preds=int8_preds,
        teacher_preds=teacher_preds,
        student_confidence=int8_conf_cal,
        tau=tau,
    )

    teacher_acc = accuracy_score(test_labels, teacher_preds)
    teacher_f1 = f1_score(test_labels, teacher_preds, average="macro", zero_division=0)
    hybrid_acc = accuracy_score(test_labels, hybrid_preds)
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

    print("=== Hybrid runtime ===")
    n = len(test_texts)
    t0 = time.time()
    rt_int8_preds, _, rt_int8_logits, student_stage_time = predict_transformer_cpu(
        int8_model,
        student.tokenizer,
        test_texts,
        max_length=args.student_max_length,
        batch_size=32,
    )
    rt_probs = temp_scaler.calibrate(rt_int8_logits)
    rt_conf = np.max(rt_probs, axis=1)
    deferred_idx = np.where(rt_conf < tau)[0]
    deferred_texts = [test_texts[i] for i in deferred_idx.tolist()]

    teacher_stage_t0 = time.time()
    deferred_teacher_preds = np.array([], dtype=object)
    if len(deferred_texts) > 0:
        deferred_teacher_preds = np.array(
            [r[0] for r in teacher.analyze_batch(deferred_texts, batch_size=32)],
            dtype=object,
        )
    teacher_stage_time = time.time() - teacher_stage_t0
    _ = t0  # keep explicit staging variable usage for traceability

    rt_hybrid = rt_int8_preds.copy()
    if len(deferred_idx) > 0:
        rt_hybrid[deferred_idx] = deferred_teacher_preds
    hybrid_total_time = student_stage_time + teacher_stage_time

    results = {
        "version": "v3_quantized_student",
        "config": vars(args),
        "training": {
            "ce_history": ce_history,
            "kd_enabled": bool(args.use_kd),
            "kd_history": kd_history,
        },
        "threshold_optimization": {
            "validation_teacher_acc": float(val_teacher_acc),
            "opt_results": {str(k): v for k, v in opt_results.items()},
            "target_operating_point": target,
            "selected_tau": float(tau),
        },
        "quantization": {
            "fp32_cpu_accuracy": float(accuracy_score(test_labels, fp32_preds)),
            "int8_accuracy": float(accuracy_score(test_labels, int8_preds)),
            "accuracy_delta_int8_minus_fp32": float(
                accuracy_score(test_labels, int8_preds)
                - accuracy_score(test_labels, fp32_preds)
            ),
            "fp32_cpu_time_s": float(fp32_time),
            "int8_time_s": float(int8_time),
            "int8_speedup_vs_fp32": float(fp32_time / max(int8_time, 1e-9)),
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
                "total_time": float(teacher_time),
                "throughput": float(n / max(teacher_time, 1e-9)),
            },
            "DistilBERT-FP32-CPU": {
                "total_time": float(fp32_time),
                "throughput": float(n / max(fp32_time, 1e-9)),
            },
            "DistilBERT-INT8-CPU": {
                "total_time": float(int8_time),
                "throughput": float(n / max(int8_time, 1e-9)),
            },
            "Hybrid-v3-int8": {
                "total_time": float(hybrid_total_time),
                "throughput": float(n / max(hybrid_total_time, 1e-9)),
                "student_stage_time": float(student_stage_time),
                "teacher_stage_time": float(teacher_stage_time),
                "teacher_usage_pct": float((len(deferred_idx) / max(1, n)) * 100.0),
            },
        },
    }

    out_json = out_dir / "all_results_v3_quantized_student.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
