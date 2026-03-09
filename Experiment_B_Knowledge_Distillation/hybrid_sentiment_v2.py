"""
Hybrid sentiment v2:
- Student: fine-tuned DistilBERT (optional KD with FinBERT teacher logits)
- Teacher: FinBERT
- Router: calibrated student confidence threshold

This file is intentionally separate from `hybrid_sentiment.py` so prior
experiments remain intact as evidence.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)



import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_sentiment import (
    CostAwareOptimizer,
    FinBERTAnalyzer,
    TemperatureScaler,
    VADERAnalyzer,
)
from utils import (
    DEVICE,
    ID2LABEL,
    LABEL2ID,
    LABEL_LIST,
    load_all_phrasebank,
    set_seed,
    split_data,
)


@dataclass
class TrainConfig:
    ce_epochs: int = 4
    kd_epochs: int = 2
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    alpha: float = 0.5
    kd_temperature: float = 2.0


class PhraseBankDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer,
        max_length: int = 128,
        teacher_logits: Optional[np.ndarray] = None,
    ):
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor([LABEL2ID[l] for l in labels], dtype=torch.long)
        self.teacher_logits = None
        if teacher_logits is not None:
            self.teacher_logits = torch.tensor(teacher_logits, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
        if self.teacher_logits is not None:
            item["teacher_logits"] = self.teacher_logits[idx]
        return item


class DistilStudent:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)

    def _make_loader(
        self,
        texts: List[str],
        labels: List[str],
        batch_size: int,
        shuffle: bool,
        teacher_logits: Optional[np.ndarray] = None,
    ) -> DataLoader:
        ds = PhraseBankDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            teacher_logits=teacher_logits,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _evaluate_loader(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"].numpy()
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k not in ("labels", "teacher_logits")
                }
                logits = self.model(**inputs).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"accuracy": float(acc), "f1_macro": float(f1)}

    def train_ce(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        cfg: TrainConfig,
    ) -> List[Dict[str, float]]:
        train_loader = self._make_loader(
            train_texts, train_labels, batch_size=cfg.batch_size, shuffle=True
        )
        val_loader = self._make_loader(
            val_texts, val_labels, batch_size=cfg.batch_size, shuffle=False
        )

        optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps = max(1, len(train_loader) * cfg.ce_epochs)
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_state = None
        best_f1 = -1.0
        history = []

        for epoch in range(cfg.ce_epochs):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"CE epoch {epoch+1}/{cfg.ce_epochs}", leave=False):
                labels = batch["labels"].to(self.device)
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k not in ("labels", "teacher_logits")
                }
                out = self.model(**inputs)
                loss = F.cross_entropy(out.logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            val_metrics = self._evaluate_loader(val_loader)
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": float(total_loss / max(1, len(train_loader))),
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
            history.append(epoch_result)
            print(
                f"CE epoch {epoch+1}: loss={epoch_result['train_loss']:.4f}, "
                f"val_acc={epoch_result['val_accuracy']:.4f}, val_f1={epoch_result['val_f1_macro']:.4f}"
            )
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return history

    def train_kd(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        train_teacher_logits: np.ndarray,
        val_teacher_logits: np.ndarray,
        cfg: TrainConfig,
    ) -> List[Dict[str, float]]:
        train_loader = self._make_loader(
            train_texts,
            train_labels,
            batch_size=cfg.batch_size,
            shuffle=True,
            teacher_logits=train_teacher_logits,
        )
        val_loader = self._make_loader(
            val_texts, val_labels, batch_size=cfg.batch_size, shuffle=False
        )

        optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps = max(1, len(train_loader) * cfg.kd_epochs)
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_state = None
        best_f1 = -1.0
        history = []

        for epoch in range(cfg.kd_epochs):
            self.model.train()
            ce_loss_sum = 0.0
            kd_loss_sum = 0.0
            total_loss_sum = 0.0

            for batch in tqdm(train_loader, desc=f"KD epoch {epoch+1}/{cfg.kd_epochs}", leave=False):
                labels = batch["labels"].to(self.device)
                teacher_logits = batch["teacher_logits"].to(self.device)
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k not in ("labels", "teacher_logits")
                }
                out = self.model(**inputs)
                student_logits = out.logits

                ce = F.cross_entropy(student_logits, labels)
                t = cfg.kd_temperature
                kd = F.kl_div(
                    F.log_softmax(student_logits / t, dim=-1),
                    F.softmax(teacher_logits / t, dim=-1),
                    reduction="batchmean",
                ) * (t**2)
                loss = cfg.alpha * ce + (1.0 - cfg.alpha) * kd

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                ce_loss_sum += ce.item()
                kd_loss_sum += kd.item()
                total_loss_sum += loss.item()

            val_metrics = self._evaluate_loader(val_loader)
            epoch_result = {
                "epoch": epoch + 1,
                "train_ce_loss": float(ce_loss_sum / max(1, len(train_loader))),
                "train_kd_loss": float(kd_loss_sum / max(1, len(train_loader))),
                "train_total_loss": float(total_loss_sum / max(1, len(train_loader))),
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
            history.append(epoch_result)
            print(
                f"KD epoch {epoch+1}: total={epoch_result['train_total_loss']:.4f}, "
                f"ce={epoch_result['train_ce_loss']:.4f}, kd={epoch_result['train_kd_loss']:.4f}, "
                f"val_acc={epoch_result['val_accuracy']:.4f}, val_f1={epoch_result['val_f1_macro']:.4f}"
            )
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return history

    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        preds = []
        confs = []
        logits_out = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
            logits_out.append(logits.cpu().numpy())
            confs.extend(conf.cpu().numpy().tolist())
            preds.extend([ID2LABEL[int(p)] for p in pred.cpu().numpy().tolist()])

        return np.array(preds), np.array(confs), np.vstack(logits_out)


def reorder_teacher_logits(logits: np.ndarray, label_map: Dict[int, str]) -> np.ndarray:
    label_to_idx = {lbl.lower(): int(idx) for idx, lbl in label_map.items()}
    indices = [label_to_idx[lbl] for lbl in LABEL_LIST]
    return logits[:, indices]


def route_predictions(
    student_preds: np.ndarray, teacher_preds: np.ndarray, student_confidence: np.ndarray, tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    use_student = student_confidence >= tau
    final_preds = np.where(use_student, student_preds, teacher_preds)
    return final_preds, use_student


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


def plot_baseline_comparison(exp: Dict, out_dir: Path):
    models = list(exp.keys())
    accs = [exp[m]["accuracy"] for m in models]
    times = [exp[m]["avg_time_ms"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(models, accs, color=["steelblue", "seagreen", "coral"])
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.0)

    axes[1].bar(models, times, color=["steelblue", "seagreen", "coral"])
    axes[1].set_title("Avg Latency")
    axes[1].set_ylabel("Time per sample (ms)")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_dir / "01_v2_baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_tradeoff(sweep: List[Dict], out_dir: Path):
    taus = [x["tau"] for x in sweep]
    ret = [x["retention"] for x in sweep]
    teacher_pct = [x["teacher_pct"] for x in sweep]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(taus, ret, color="teal", linewidth=2, label="Accuracy retention")
    ax1.set_xlabel("Confidence threshold (tau)")
    ax1.set_ylabel("Retention vs teacher", color="teal")
    ax1.tick_params(axis="y", labelcolor="teal")
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    ax2.plot(taus, teacher_pct, color="darkred", linewidth=2, label="Teacher usage %")
    ax2.set_ylabel("Teacher usage (%)", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")

    plt.title("Retention vs Teacher Usage")
    plt.tight_layout()
    plt.savefig(out_dir / "02_v2_threshold_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_efficiency(exp: Dict, out_dir: Path):
    models = ["FinBERT", "DistilBERT", "Hybrid-v2"]
    throughput = [exp[m]["throughput"] for m in models]
    time_total = [exp[m]["total_time"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(models, throughput, color=["coral", "seagreen", "slateblue"])
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("Samples / sec")

    axes[1].bar(models, time_total, color=["coral", "seagreen", "slateblue"])
    axes[1].set_title("Total Time")
    axes[1].set_ylabel("Seconds")

    plt.tight_layout()
    plt.savefig(out_dir / "03_v2_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_agreement_tier(exp: Dict, out_dir: Path):
    tiers = ["100", "75", "66", "50"]
    x = np.arange(len(tiers))
    width = 0.25

    student_acc = [exp[t]["student_acc"] for t in tiers]
    teacher_acc = [exp[t]["teacher_acc"] for t in tiers]
    hybrid_acc = [exp[t]["hybrid_acc"] for t in tiers]
    teacher_pct = [exp[t]["teacher_pct"] for t in tiers]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(x - width, student_acc, width, label="DistilBERT", color="seagreen")
    axes[0].bar(x, teacher_acc, width, label="FinBERT", color="coral")
    axes[0].bar(x + width, hybrid_acc, width, label="Hybrid-v2", color="slateblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tiers)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xlabel("Agreement Tier")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_title("Per-tier Accuracy")

    axes[1].bar(tiers, teacher_pct, color="darkred")
    axes[1].set_xlabel("Agreement Tier")
    axes[1].set_ylabel("Teacher usage (%)")
    axes[1].set_title("Per-tier Teacher Usage")

    plt.tight_layout()
    plt.savefig(out_dir / "04_v2_agreement_tier.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid sentiment v2 (DistilBERT student + FinBERT teacher)"
    )
    _root = str(Path(__file__).resolve().parent.parent)
    parser.add_argument("--data-dir", type=str, default=str(Path(_root) / "FinancialPhraseBank-v1.0"))
    parser.add_argument("--results-dir", type=str, default="results_v2")
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

    print(f"Device: {DEVICE}")
    print(f"Results directory: {out_dir}")

    print("\n=== Data ===")
    df = load_all_phrasebank(args.data_dir)
    train_df, val_df, test_df = split_data(df, test_size=args.test_split, val_size=args.val_split)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print("\n=== Models ===")
    student = DistilStudent(model_name=args.student_model, max_length=args.student_max_length)
    teacher = FinBERTAnalyzer()
    vader = VADERAnalyzer()

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

    print("\n=== Phase 1: CE fine-tuning ===")
    ce_history = student.train_ce(train_texts, train_labels, val_texts, val_labels, cfg)

    kd_history = []
    if args.use_kd:
        print("\n=== Phase 2: KD training ===")
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

    print("\n=== Calibration + Routing setup ===")
    val_student_preds, _, val_student_logits = student.predict_batch(val_texts, batch_size=32)
    val_teacher_res = teacher.analyze_batch(val_texts, batch_size=32)
    val_teacher_preds = np.array([r[0] for r in val_teacher_res])
    val_labels_arr = np.array(val_labels)

    temp_scaler = TemperatureScaler()
    val_label_ids = np.array([LABEL2ID[l] for l in val_labels])
    temp_scaler.fit(val_student_logits, val_label_ids)
    val_cal_probs = temp_scaler.calibrate(val_student_logits)
    val_cal_conf = np.max(val_cal_probs, axis=1)

    val_teacher_acc = accuracy_score(val_labels_arr, val_teacher_preds)
    optimizer = CostAwareOptimizer()
    opt_results = optimizer.optimize(
        val_cal_conf,
        val_student_preds,
        val_teacher_preds,
        val_labels_arr,
    )

    op_target = optimizer.find_target_operating_point(
        val_cal_conf,
        val_student_preds,
        val_teacher_preds,
        val_labels_arr,
        finbert_acc=val_teacher_acc,
        target_retention=args.target_retention,
        max_finbert_pct=args.max_teacher_pct,
    )

    if op_target is not None and op_target["meets_target"]:
        optimal_tau = float(op_target["tau"])
    else:
        fallback = min(opt_results.items(), key=lambda kv: kv[1]["cost"])
        optimal_tau = float(fallback[1]["tau"])

    print(f"Selected threshold tau={optimal_tau:.3f}")

    print("\n=== Experiment A: New baseline ===")
    t0 = time.time()
    vader_res = vader.analyze_batch(test_texts)
    vader_time = time.time() - t0
    vader_preds = np.array([r[0] for r in vader_res])

    t0 = time.time()
    student_preds, _, student_logits = student.predict_batch(test_texts, batch_size=32)
    student_time = time.time() - t0
    student_probs_cal = temp_scaler.calibrate(student_logits)
    student_conf_cal = np.max(student_probs_cal, axis=1)

    t0 = time.time()
    teacher_res = teacher.analyze_batch(test_texts, batch_size=32)
    teacher_time = time.time() - t0
    teacher_preds = np.array([r[0] for r in teacher_res])

    n_test = len(test_texts)
    exp_a = {
        "VADER": {
            "accuracy": float(accuracy_score(test_labels, vader_preds)),
            "f1_macro": float(f1_score(test_labels, vader_preds, average="macro", zero_division=0)),
            "total_time": float(vader_time),
            "avg_time_ms": float((vader_time / n_test) * 1000),
        },
        "DistilBERT": {
            "accuracy": float(accuracy_score(test_labels, student_preds)),
            "f1_macro": float(f1_score(test_labels, student_preds, average="macro", zero_division=0)),
            "total_time": float(student_time),
            "avg_time_ms": float((student_time / n_test) * 1000),
        },
        "FinBERT": {
            "accuracy": float(accuracy_score(test_labels, teacher_preds)),
            "f1_macro": float(f1_score(test_labels, teacher_preds, average="macro", zero_division=0)),
            "total_time": float(teacher_time),
            "avg_time_ms": float((teacher_time / n_test) * 1000),
        },
    }

    print("\n=== Experiment B: Hypothesis re-test ===")
    hybrid_preds, use_student = route_predictions(
        student_preds=student_preds,
        teacher_preds=teacher_preds,
        student_confidence=student_conf_cal,
        tau=optimal_tau,
    )
    teacher_usage_pct = float((~use_student).mean() * 100.0)
    teacher_acc = accuracy_score(test_labels, teacher_preds)
    hybrid_acc = accuracy_score(test_labels, hybrid_preds)
    teacher_f1 = f1_score(test_labels, teacher_preds, average="macro", zero_division=0)
    hybrid_f1 = f1_score(test_labels, hybrid_preds, average="macro", zero_division=0)
    acc_retention = hybrid_acc / teacher_acc if teacher_acc > 0 else 0.0
    f1_retention = hybrid_f1 / teacher_f1 if teacher_f1 > 0 else 0.0

    ci = bootstrap_hypothesis_ci(
        labels=np.array(test_labels),
        hybrid_preds=hybrid_preds,
        teacher_preds=teacher_preds,
        use_student=use_student,
        n_bootstrap=args.n_bootstrap,
    )
    mcn = mcnemar_test(
        hybrid_correct=(hybrid_preds == np.array(test_labels)),
        teacher_correct=(teacher_preds == np.array(test_labels)),
    )

    sweep = []
    for tau in np.linspace(0.0, 1.0, 51):
        preds_tau, use_student_tau = route_predictions(
            student_preds, teacher_preds, student_conf_cal, float(tau)
        )
        acc_tau = accuracy_score(test_labels, preds_tau)
        ret_tau = acc_tau / teacher_acc if teacher_acc > 0 else 0.0
        sweep.append(
            {
                "tau": float(tau),
                "accuracy": float(acc_tau),
                "retention": float(ret_tau),
                "teacher_pct": float((~use_student_tau).mean() * 100.0),
            }
        )

    exp_b = {
        "teacher_accuracy": float(teacher_acc),
        "teacher_f1": float(teacher_f1),
        "hybrid_accuracy": float(hybrid_acc),
        "hybrid_f1": float(hybrid_f1),
        "accuracy_retention": float(acc_retention),
        "f1_retention": float(f1_retention),
        "teacher_usage_pct": float(teacher_usage_pct),
        "target_retention": float(args.target_retention),
        "target_max_teacher_pct": float(args.max_teacher_pct),
        "hypothesis_accuracy_met": bool(acc_retention >= args.target_retention),
        "hypothesis_cost_met": bool(teacher_usage_pct <= args.max_teacher_pct),
        "selected_tau": float(optimal_tau),
        "ci_retention_95": ci["retention_ci_95"],
        "ci_teacher_usage_95": ci["teacher_usage_ci_95"],
        "mcnemar_statistic": mcn["statistic"],
        "mcnemar_p_value": mcn["p_value"],
        "sweep": sweep,
    }

    print("\n=== Experiment C: Efficiency analysis ===")
    t0 = time.time()
    student_preds_rt, _, student_logits_rt = student.predict_batch(test_texts, batch_size=32)
    student_probs_rt = temp_scaler.calibrate(student_logits_rt)
    student_conf_rt = np.max(student_probs_rt, axis=1)
    route_time = time.time() - t0

    deferred_idx = np.where(student_conf_rt < optimal_tau)[0]
    deferred_texts = [test_texts[i] for i in deferred_idx.tolist()]

    t0 = time.time()
    deferred_teacher_preds = np.array([], dtype=object)
    if len(deferred_texts) > 0:
        deferred_teacher_res = teacher.analyze_batch(deferred_texts, batch_size=32)
        deferred_teacher_preds = np.array([r[0] for r in deferred_teacher_res], dtype=object)
    deferred_teacher_time = time.time() - t0

    hybrid_rt_preds = student_preds_rt.copy()
    if len(deferred_idx) > 0:
        hybrid_rt_preds[deferred_idx] = deferred_teacher_preds
    hybrid_total_time = route_time + deferred_teacher_time

    formula_total = route_time + ((~(student_conf_rt >= optimal_tau)).mean() * teacher_time)
    exp_c = {
        "FinBERT": {
            "total_time": float(teacher_time),
            "throughput": float(n_test / max(teacher_time, 1e-9)),
        },
        "DistilBERT": {
            "total_time": float(student_time),
            "throughput": float(n_test / max(student_time, 1e-9)),
        },
        "Hybrid-v2": {
            "total_time": float(hybrid_total_time),
            "throughput": float(n_test / max(hybrid_total_time, 1e-9)),
            "teacher_usage_pct": float((len(deferred_idx) / max(1, n_test)) * 100.0),
            "student_stage_time": float(route_time),
            "teacher_stage_time": float(deferred_teacher_time),
            "formula_time_estimate": float(formula_total),
        },
    }

    print("\n=== Experiment D: Agreement-tier breakdown ===")
    exp_d = {}
    test_tiers = test_df["agreement_tier"].tolist()
    for tier in ["100", "75", "66", "50"]:
        idx = np.array([i for i, t in enumerate(test_tiers) if t == tier], dtype=int)
        if len(idx) == 0:
            exp_d[tier] = {
                "n": 0,
                "student_acc": 0.0,
                "teacher_acc": 0.0,
                "hybrid_acc": 0.0,
                "teacher_pct": 0.0,
            }
            continue
        y = np.array(test_labels)[idx]
        exp_d[tier] = {
            "n": int(len(idx)),
            "student_acc": float(accuracy_score(y, student_preds[idx])),
            "teacher_acc": float(accuracy_score(y, teacher_preds[idx])),
            "hybrid_acc": float(accuracy_score(y, hybrid_preds[idx])),
            "teacher_pct": float((~use_student[idx]).mean() * 100.0),
        }

    results = {
        "version": "v2_distilbert_student",
        "config": vars(args),
        "training": {
            "ce_history": ce_history,
            "kd_enabled": bool(args.use_kd),
            "kd_history": kd_history,
        },
        "threshold_optimization": {
            "validation_teacher_acc": float(val_teacher_acc),
            "opt_results": {str(k): v for k, v in opt_results.items()},
            "target_operating_point": op_target,
            "selected_tau": float(optimal_tau),
        },
        "expA_baselines": exp_a,
        "expB_hypothesis": exp_b,
        "expC_efficiency": exp_c,
        "expD_agreement_tier": exp_d,
    }

    plot_baseline_comparison(exp_a, out_dir)
    plot_threshold_tradeoff(exp_b["sweep"], out_dir)
    plot_efficiency(exp_c, out_dir)
    plot_agreement_tier(exp_d, out_dir)

    out_json = out_dir / "all_results_v2.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)

    print("\n=== Final summary ===")
    print(f"Teacher accuracy:    {exp_b['teacher_accuracy']:.4f}")
    print(f"Hybrid-v2 accuracy:  {exp_b['hybrid_accuracy']:.4f}")
    print(f"Accuracy retention:  {exp_b['accuracy_retention']:.4f}")
    print(f"Teacher usage (%):   {exp_b['teacher_usage_pct']:.2f}")
    print(f"Selected tau:        {exp_b['selected_tau']:.3f}")
    print(
        f"Hypothesis met:      {exp_b['hypothesis_accuracy_met'] and exp_b['hypothesis_cost_met']}"
    )
    print(f"Saved results to:    {out_json}")
    print(f"Saved plots to:      {out_dir}")


if __name__ == "__main__":
    main()
