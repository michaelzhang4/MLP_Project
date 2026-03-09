"""
Variant 4: Quantized early-exit FinBERT with entropy-based routing.

Architecture:
- Single FinBERT backbone (INT8 dynamic quantization on CPU)
- Stage 1 runs layers 1-4 and produces early logits via a LogisticRegression head
- Entropy of softmax(early_logits) determines confidence:
  low entropy = confident = exit early; high entropy = uncertain = continue deep
- Threshold (tau) on entropy is selected via validation sweep
"""

import argparse
import copy
import io
import json
import sys
import time
from pathlib import Path

# Allow imports from the project root and parent experiment folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.ao.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import (
    LABEL_LIST,
    bootstrap_hypothesis_ci,
    load_all_phrasebank,
    make_json_safe,
    mcnemar_test,
    set_seed,
    split_data,
)


DEFAULT_DEMO_TEXT = "Revenue fell, but strategic guidance remains strong."


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1e-12, None)


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Per-sample entropy of probability vectors. Lower = more confident."""
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def _logits_to_labels(logits: np.ndarray) -> np.ndarray:
    return np.array([LABEL_LIST[i] for i in np.argmax(logits, axis=1)], dtype=object)


def _decision_function_logits(
    early_head: LogisticRegression, cls_feats: np.ndarray
) -> np.ndarray:
    logits = early_head.decision_function(cls_feats)
    if logits.ndim == 1:
        raise RuntimeError("Early-exit head must be multiclass for LABEL_LIST.")

    classes = [str(c).lower() for c in early_head.classes_]
    reorder = [classes.index(lbl) for lbl in LABEL_LIST]
    return logits[:, reorder]


def _serialized_model_size_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return float(len(buffer.getbuffer()) / (1024.0 * 1024.0))


def _select_tau(
    entropy_scores: np.ndarray,
    early_preds: np.ndarray,
    full_preds: np.ndarray,
    labels: np.ndarray,
    exit_layer: int,
    num_hidden_layers: int,
    target_retention: float,
    max_layer_fraction: float,
    cost_lambda: float,
    tau_min: float,
    tau_max: float,
    tau_steps: int,
) -> Tuple[float, Dict[str, float], List[Dict[str, float]], float]:
    """Sweep entropy threshold tau. Samples with entropy <= tau exit early."""
    full_acc = accuracy_score(labels, full_preds)
    sweep = []
    for tau in np.linspace(tau_min, tau_max, tau_steps):
        use_early = entropy_scores <= float(tau)
        preds = np.where(use_early, early_preds, full_preds)
        acc = accuracy_score(labels, preds)
        retention = acc / full_acc if full_acc > 0 else 0.0
        avg_layers = (use_early.mean() * exit_layer) + ((~use_early).mean() * num_hidden_layers)
        layer_fraction = avg_layers / float(num_hidden_layers)
        cost = (1.0 - acc) + (cost_lambda * layer_fraction)
        sweep.append(
            {
                "tau": float(tau),
                "accuracy": float(acc),
                "retention": float(retention),
                "early_exit_pct": float(use_early.mean() * 100.0),
                "avg_layers": float(avg_layers),
                "layer_fraction": float(layer_fraction),
                "cost": float(cost),
            }
        )

    feasible = [
        s
        for s in sweep
        if s["retention"] >= target_retention and s["layer_fraction"] <= max_layer_fraction
    ]
    if feasible:
        best = sorted(feasible, key=lambda x: (-x["retention"], -x["accuracy"], x["layer_fraction"]))[0]
    else:
        best = min(sweep, key=lambda x: x["cost"])

    return float(best["tau"]), best, sweep, float(full_acc)


class QuantizedEarlyExitFinBERT:
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        exit_layer: int = 4,
        max_length: int = 128,
    ):
        self.device = torch.device("cpu")
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.fp32_model = AutoModelForSequenceClassification.from_pretrained(model_name).cpu().eval()
        self.model = quantize_dynamic(
            copy.deepcopy(self.fp32_model).cpu().eval(),
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        self.model.eval()

        self.num_hidden_layers = int(self.model.config.num_hidden_layers)
        if not (1 <= exit_layer < self.num_hidden_layers):
            raise ValueError(
                f"exit_layer must be in [1, {self.num_hidden_layers - 1}], got {exit_layer}"
            )
        self.exit_layer = int(exit_layer)

        model_idx_to_label = {
            int(idx): str(lbl).lower() for idx, lbl in self.model.config.id2label.items()
        }
        label_to_model_idx = {lbl: idx for idx, lbl in model_idx_to_label.items()}
        self.reorder_idx = [label_to_model_idx[lbl] for lbl in LABEL_LIST]

    def _tokenize_batch(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _forward_stage1(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))

        bert = self.model.bert
        hidden_states = bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attention_mask = self.model.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        for layer_idx in range(self.exit_layer):
            layer_outputs = bert.encoder.layer[layer_idx](
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = layer_outputs[0]

        return hidden_states, extended_attention_mask

    def _forward_stage2(
        self, hidden_states: torch.Tensor, extended_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bert = self.model.bert
        for layer_idx in range(self.exit_layer, self.num_hidden_layers):
            layer_outputs = bert.encoder.layer[layer_idx](
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = layer_outputs[0]

        if bert.pooler is not None:
            pooled = bert.pooler(hidden_states)
        else:
            pooled = hidden_states[:, 0, :]
        pooled = self.model.dropout(pooled)
        logits = self.model.classifier(pooled)
        return logits[:, self.reorder_idx]

    def _forward_full_logits(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
            return_dict=True,
        ).logits
        return logits[:, self.reorder_idx]

    def extract_exit_cls_features(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        cls_list = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self._tokenize_batch(batch)
                hidden_states, _ = self._forward_stage1(inputs)
                cls = hidden_states[:, 0, :].cpu().numpy()
                cls_list.append(cls)
        return np.vstack(cls_list)

    def predict_full(
        self, texts: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        preds = []
        logits_all = []

        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self._tokenize_batch(batch)
                logits = self._forward_full_logits(inputs)
                logits_np = logits.cpu().numpy()
                logits_all.append(logits_np)
                preds.extend(_logits_to_labels(logits_np).tolist())
        elapsed = time.time() - t0
        return np.array(preds, dtype=object), np.vstack(logits_all), float(elapsed)

    def benchmark_hybrid_runtime(
        self,
        texts: List[str],
        early_head: LogisticRegression,
        tau: float,
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray | float]:
        all_preds = []
        used_layers = []

        stage1_time = 0.0
        router_time = 0.0
        stage2_time = 0.0

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self._tokenize_batch(batch)

                t0 = time.time()
                hidden_states, extended_attention_mask = self._forward_stage1(inputs)
                stage1_time += time.time() - t0

                cls_feats = hidden_states[:, 0, :].cpu().numpy()
                early_logits = _decision_function_logits(early_head, cls_feats)
                early_preds = _logits_to_labels(early_logits)

                t0 = time.time()
                probs = _softmax_np(early_logits)
                ent = _entropy(probs)
                use_early = ent <= tau
                router_time += time.time() - t0

                batch_preds = early_preds.copy()
                batch_layers = np.full(len(batch), self.num_hidden_layers, dtype=int)
                batch_layers[use_early] = self.exit_layer

                if np.any(~use_early):
                    t0 = time.time()
                    deep_logits = self._forward_stage2(
                        hidden_states[~use_early], extended_attention_mask[~use_early]
                    )
                    stage2_time += time.time() - t0
                    deep_preds = _logits_to_labels(deep_logits.cpu().numpy())
                    batch_preds[~use_early] = deep_preds

                all_preds.extend(batch_preds.tolist())
                used_layers.extend(batch_layers.tolist())

        total_time = stage1_time + router_time + stage2_time
        return {
            "preds": np.array(all_preds, dtype=object),
            "used_layers": np.array(used_layers, dtype=int),
            "stage1_time": float(stage1_time),
            "router_time": float(router_time),
            "stage2_time": float(stage2_time),
            "total_time": float(total_time),
        }


def _run_demo_case(
    model: QuantizedEarlyExitFinBERT,
    early_head: LogisticRegression,
    tau: float,
    text: str,
) -> Dict[str, object]:
    with torch.no_grad():
        inputs = model._tokenize_batch([text])
        hidden_states, extended_attention_mask = model._forward_stage1(inputs)
        cls_feats = hidden_states[:, 0, :].cpu().numpy()
        early_logits = _decision_function_logits(early_head, cls_feats)
        early_probs = _softmax_np(early_logits)
        early_pred = _logits_to_labels(early_logits)[0]

        ent = float(_entropy(early_probs)[0])
        use_early = ent <= tau

        final_pred = early_pred
        if not use_early:
            deep_logits = model._forward_stage2(hidden_states, extended_attention_mask)
            final_pred = _logits_to_labels(deep_logits.cpu().numpy())[0]

    return {
        "text": text,
        "layer4_logits": [float(x) for x in early_logits[0].tolist()],
        "layer4_probs": [float(x) for x in early_probs[0].tolist()],
        "layer4_prediction": str(early_pred),
        "entropy": ent,
        "tau": float(tau),
        "router_decision": "EXIT_EARLY" if use_early else "CONTINUE_DEEP",
        "final_prediction": str(final_pred),
        "path_layers": f"1-{model.exit_layer}" if use_early else f"1-{model.num_hidden_layers}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid v4: Quantized Early-Exit FinBERT with entropy-based routing"
    )
    parser.add_argument("--data-dir", type=str, default="FinancialPhraseBank-v1.0")
    parser.add_argument(
        "--results-dir", type=str, default="results_v4_quantized_early_exit_router"
    )
    parser.add_argument("--model-name", type=str, default="ProsusAI/finbert")
    parser.add_argument("--exit-layer", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.2)

    parser.add_argument("--target-retention", type=float, default=0.98)
    parser.add_argument("--max-layer-fraction", type=float, default=0.35)
    parser.add_argument("--cost-lambda", type=float, default=0.35)
    parser.add_argument("--tau-min", type=float, default=0.01)
    parser.add_argument("--tau-max", type=float, default=1.10)
    parser.add_argument("--tau-steps", type=int, default=95)

    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--demo-text", type=str, default=DEFAULT_DEMO_TEXT)
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
    train_labels = np.array(train_df["label"].tolist(), dtype=object)
    val_texts = val_df["text"].tolist()
    val_labels = np.array(val_df["label"].tolist(), dtype=object)
    test_texts = test_df["text"].tolist()
    test_labels = np.array(test_df["label"].tolist(), dtype=object)

    print("=== Quantized FinBERT backbone ===")
    model = QuantizedEarlyExitFinBERT(
        model_name=args.model_name,
        exit_layer=args.exit_layer,
        max_length=args.max_length,
    )
    print(
        f"Model ready: exit_layer={model.exit_layer}, total_layers={model.num_hidden_layers}"
    )

    print("=== Train layer-4 early-exit head ===")
    train_cls = model.extract_exit_cls_features(train_texts, batch_size=args.batch_size)
    early_head = LogisticRegression(
        max_iter=1200,
        solver="lbfgs",
    )
    early_head.fit(train_cls, train_labels)
    train_early_logits = _decision_function_logits(early_head, train_cls)
    train_early_preds = _logits_to_labels(train_early_logits)
    train_early_acc = accuracy_score(train_labels, train_early_preds)
    print(f"Early-exit head train accuracy={train_early_acc:.4f}")

    print("=== Compute entropy scores for routing ===")
    train_early_probs = _softmax_np(train_early_logits)
    train_entropy = _entropy(train_early_probs)
    print(f"Train entropy: mean={train_entropy.mean():.4f}, std={train_entropy.std():.4f}, "
          f"min={train_entropy.min():.4f}, max={train_entropy.max():.4f}")

    print("=== Validation threshold search ===")
    val_cls = model.extract_exit_cls_features(val_texts, batch_size=args.batch_size)
    val_early_logits = _decision_function_logits(early_head, val_cls)
    val_early_preds = _logits_to_labels(val_early_logits)
    val_early_probs = _softmax_np(val_early_logits)
    val_entropy = _entropy(val_early_probs)
    val_full_preds, _, _ = model.predict_full(val_texts, batch_size=args.batch_size)

    tau, best_point, sweep, val_full_acc = _select_tau(
        entropy_scores=val_entropy,
        early_preds=val_early_preds,
        full_preds=val_full_preds,
        labels=val_labels,
        exit_layer=model.exit_layer,
        num_hidden_layers=model.num_hidden_layers,
        target_retention=args.target_retention,
        max_layer_fraction=args.max_layer_fraction,
        cost_lambda=args.cost_lambda,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_steps=args.tau_steps,
    )
    print(f"Selected tau={tau:.3f}")

    print("=== Test metrics ===")
    test_cls = model.extract_exit_cls_features(test_texts, batch_size=args.batch_size)
    test_early_logits = _decision_function_logits(early_head, test_cls)
    test_early_preds = _logits_to_labels(test_early_logits)
    test_early_probs = _softmax_np(test_early_logits)
    test_entropy = _entropy(test_early_probs)
    full_preds, full_logits, full_time = model.predict_full(
        test_texts, batch_size=args.batch_size
    )

    use_early = test_entropy <= tau
    hybrid_preds = np.where(use_early, test_early_preds, full_preds)

    full_acc = accuracy_score(test_labels, full_preds)
    full_f1 = f1_score(test_labels, full_preds, average="macro", zero_division=0)
    hybrid_acc = accuracy_score(test_labels, hybrid_preds)
    hybrid_f1 = f1_score(test_labels, hybrid_preds, average="macro", zero_division=0)
    retention = hybrid_acc / full_acc if full_acc > 0 else 0.0

    used_layers = np.where(use_early, model.exit_layer, model.num_hidden_layers)
    avg_layers = float(np.mean(used_layers))
    layer_fraction = float(avg_layers / model.num_hidden_layers)
    approx_speedup = float(1.0 / max(layer_fraction, 1e-9))
    early_exit_pct = float(np.mean(use_early) * 100.0)
    deferred_pct = float(np.mean(~use_early) * 100.0)

    ci = bootstrap_hypothesis_ci(
        labels=test_labels,
        hybrid_preds=hybrid_preds,
        teacher_preds=full_preds,
        use_student=use_early,
        n_bootstrap=args.n_bootstrap,
    )
    mcn = mcnemar_test(
        hybrid_correct=(hybrid_preds == test_labels),
        teacher_correct=(full_preds == test_labels),
    )

    print("=== Runtime benchmark (staged execution) ===")
    staged = model.benchmark_hybrid_runtime(
        test_texts,
        early_head=early_head,
        tau=tau,
        batch_size=args.batch_size,
    )
    runtime_mismatch = int(np.sum(staged["preds"] != hybrid_preds))
    hybrid_time = float(staged["total_time"])
    runtime_speedup = float(full_time / max(hybrid_time, 1e-9))

    fp32_size_mb = _serialized_model_size_mb(model.fp32_model)
    int8_size_mb = _serialized_model_size_mb(model.model)
    int8_ratio = float(int8_size_mb / max(fp32_size_mb, 1e-9))
    int8_reduction_pct = float((1.0 - int8_ratio) * 100.0)

    demo_case = _run_demo_case(
        model=model,
        early_head=early_head,
        tau=tau,
        text=args.demo_text,
    )

    n = len(test_texts)
    results = {
        "version": "v4_quantized_early_exit_entropy_router",
        "config": vars(args),
        "model": {
            "backbone": args.model_name,
            "quantized_int8_dynamic": True,
            "exit_layer": int(model.exit_layer),
            "num_hidden_layers": int(model.num_hidden_layers),
            "routing": "entropy_threshold",
        },
        "training": {
            "early_exit_head": {
                "type": "LogisticRegression(multinomial)",
                "train_accuracy": float(train_early_acc),
            },
            "router": {
                "type": "entropy_threshold",
                "description": "Exit early when softmax entropy of early logits <= tau",
                "train_entropy_mean": float(train_entropy.mean()),
                "train_entropy_std": float(train_entropy.std()),
            },
        },
        "threshold_selection": {
            "validation_full_accuracy": float(val_full_acc),
            "selected_tau": float(tau),
            "selected_point": best_point,
            "sweep": sweep,
        },
        "test_metrics": {
            "full_accuracy": float(full_acc),
            "full_f1": float(full_f1),
            "hybrid_accuracy": float(hybrid_acc),
            "hybrid_f1": float(hybrid_f1),
            "accuracy_retention": float(retention),
            "early_exit_pct": early_exit_pct,
            "deferred_pct": deferred_pct,
            "avg_layers_used": avg_layers,
            "layer_fraction": layer_fraction,
            "approx_speedup_vs_full": approx_speedup,
            "meets_retention_target": bool(retention >= args.target_retention),
            "meets_layer_budget": bool(layer_fraction <= args.max_layer_fraction),
            "ci_retention_95": ci["retention_ci_95"],
            "ci_deferred_pct_95": ci["teacher_usage_ci_95"],
            "mcnemar_statistic": mcn["statistic"],
            "mcnemar_p_value": mcn["p_value"],
        },
        "efficiency": {
            "full_quantized_finbert_time_s": float(full_time),
            "full_quantized_finbert_throughput": float(n / max(full_time, 1e-9)),
            "hybrid_time_s": float(hybrid_time),
            "hybrid_throughput": float(n / max(hybrid_time, 1e-9)),
            "runtime_speedup_vs_full": runtime_speedup,
            "stage_times_s": {
                "stage1_layers_1_4": float(staged["stage1_time"]),
                "router": float(staged["router_time"]),
                "stage2_layers_5_12": float(staged["stage2_time"]),
            },
            "staged_avg_layers_used": float(np.mean(staged["used_layers"])),
        },
        "memory": {
            "fp32_model_size_mb": float(fp32_size_mb),
            "int8_model_size_mb": float(int8_size_mb),
            "int8_size_ratio_vs_fp32": float(int8_ratio),
            "int8_size_reduction_pct": float(int8_reduction_pct),
        },
        "demo_case": demo_case,
        "sanity": {
            "runtime_prediction_mismatch_count": runtime_mismatch,
            "test_full_logits_shape": list(full_logits.shape),
        },
    }

    out_json = out_dir / "all_results_v4_quantized_early_exit_router.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)
    print(f"Saved results to {out_json}")

    print("=== Summary ===")
    print(f"Full acc={full_acc:.4f}, Hybrid acc={hybrid_acc:.4f}, retention={retention:.4f}")
    print(f"Early exit={early_exit_pct:.2f}%, Avg layers={avg_layers:.2f}/{model.num_hidden_layers}")
    print(
        f"Speedup approx={approx_speedup:.2f}x, measured={runtime_speedup:.2f}x, "
        f"INT8 size reduction={int8_reduction_pct:.2f}%"
    )


if __name__ == "__main__":
    main()
