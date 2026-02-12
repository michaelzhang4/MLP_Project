"""
Variant 2: Early-exit FinBERT.

Single-backbone model with intermediate exit heads:
- Train linear exit heads on FinBERT layer representations (default layers 4 and 8)
- Use confidence threshold to exit early
- Fall back to final FinBERT head when early heads are uncertain
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import DEVICE, LABEL_LIST, load_all_phrasebank, make_json_safe, set_seed, split_data


class EarlyExitFinBERT:
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        exit_layers: Tuple[int, ...] = (4, 8),
        max_length: int = 128,
    ):
        self.device = DEVICE
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        num_hidden_layers = int(self.model.config.num_hidden_layers)
        self.num_hidden_layers = num_hidden_layers
        self.exit_layers = tuple(
            sorted([l for l in set(exit_layers) if 1 <= l < self.num_hidden_layers])
        )
        self.exit_heads: Dict[int, LogisticRegression] = {}

        model_idx_to_label = {
            int(idx): lbl.lower() for idx, lbl in self.model.config.id2label.items()
        }
        label_to_model_idx = {lbl: idx for idx, lbl in model_idx_to_label.items()}
        self.reorder_idx = [label_to_model_idx[lbl] for lbl in LABEL_LIST]

    def _extract_features_and_final(
        self, texts: List[str], batch_size: int = 32
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        features = {layer: [] for layer in self.exit_layers}
        final_probs_all = []

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
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = outputs.hidden_states
            for layer in self.exit_layers:
                cls = hidden_states[layer][:, 0, :].detach().cpu().numpy()
                features[layer].append(cls)

            final_probs = F.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            final_probs = final_probs[:, self.reorder_idx]
            final_probs_all.append(final_probs)

        features = {k: np.vstack(v) for k, v in features.items()}
        final_probs = np.vstack(final_probs_all)
        final_conf = np.max(final_probs, axis=1)
        final_preds = np.array([LABEL_LIST[i] for i in np.argmax(final_probs, axis=1)], dtype=object)
        return features, final_probs, final_conf, final_preds

    def fit_exit_heads(
        self, train_texts: List[str], train_labels: List[str], batch_size: int = 32
    ) -> Dict[str, float]:
        feats, _, _, _ = self._extract_features_and_final(train_texts, batch_size=batch_size)
        summary = {}
        for layer in self.exit_layers:
            clf = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
            )
            clf.fit(feats[layer], train_labels)
            self.exit_heads[layer] = clf
            train_preds = clf.predict(feats[layer])
            summary[f"layer_{layer}_train_acc"] = float(
                accuracy_score(train_labels, train_preds)
            )
        return summary

    def precompute_cache(
        self, texts: List[str], batch_size: int = 32
    ) -> Dict[str, Dict[int, np.ndarray] | np.ndarray]:
        feats, final_probs, final_conf, final_preds = self._extract_features_and_final(
            texts, batch_size=batch_size
        )
        layer_conf = {}
        layer_preds = {}
        for layer in self.exit_layers:
            clf = self.exit_heads[layer]
            probs_raw = clf.predict_proba(feats[layer])
            classes = list(clf.classes_)
            idx = [classes.index(lbl) for lbl in LABEL_LIST]
            probs = probs_raw[:, idx]
            layer_conf[layer] = np.max(probs, axis=1)
            layer_preds[layer] = np.array(
                [LABEL_LIST[i] for i in np.argmax(probs, axis=1)], dtype=object
            )
        return {
            "layer_conf": layer_conf,
            "layer_preds": layer_preds,
            "final_conf": final_conf,
            "final_preds": final_preds,
        }

    def compose_predictions(
        self, cache: Dict[str, Dict[int, np.ndarray] | np.ndarray], tau: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        final_preds = cache["final_preds"]
        n = len(final_preds)
        preds = np.empty(n, dtype=object)
        used_layers = np.full(n, self.num_hidden_layers, dtype=int)
        undecided = np.ones(n, dtype=bool)

        for layer in self.exit_layers:
            conf = cache["layer_conf"][layer]
            p = cache["layer_preds"][layer]
            take = undecided & (conf >= tau)
            preds[take] = p[take]
            used_layers[take] = layer
            undecided[take] = False

        if np.any(undecided):
            preds[undecided] = final_preds[undecided]
        return preds, used_layers


def main():
    parser = argparse.ArgumentParser(description="Hybrid v3 early-exit FinBERT")
    parser.add_argument("--data-dir", type=str, default="FinancialPhraseBank-v1.0")
    parser.add_argument("--results-dir", type=str, default="results_v3_early_exit")
    parser.add_argument("--model-name", type=str, default="ProsusAI/finbert")
    parser.add_argument("--exit-layers", type=str, default="4,8")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--target-retention", type=float, default=0.98)
    parser.add_argument("--max-layer-fraction", type=float, default=0.35)
    parser.add_argument("--cost-lambda", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(exist_ok=True)
    set_seed(args.seed)

    exit_layers = tuple(int(x.strip()) for x in args.exit_layers.split(",") if x.strip())

    print("=== Data ===")
    df = load_all_phrasebank(args.data_dir)
    train_df, val_df, test_df = split_data(
        df, test_size=args.test_split, val_size=args.val_split
    )
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = np.array(val_df["label"].tolist(), dtype=object)
    test_texts = test_df["text"].tolist()
    test_labels = np.array(test_df["label"].tolist(), dtype=object)

    print("=== Model ===")
    model = EarlyExitFinBERT(
        model_name=args.model_name,
        exit_layers=exit_layers,
        max_length=args.max_length,
    )

    print("=== Train exit heads ===")
    train_summary = model.fit_exit_heads(
        train_texts=train_texts,
        train_labels=train_labels,
        batch_size=args.batch_size,
    )

    print("=== Validation threshold search ===")
    val_cache = model.precompute_cache(val_texts, batch_size=args.batch_size)
    val_teacher_preds = val_cache["final_preds"]
    val_teacher_acc = accuracy_score(val_labels, val_teacher_preds)

    sweep = []
    for tau in np.linspace(0.5, 0.99, 50):
        preds, used_layers = model.compose_predictions(val_cache, float(tau))
        acc = accuracy_score(val_labels, preds)
        retention = acc / val_teacher_acc if val_teacher_acc > 0 else 0.0
        layer_fraction = used_layers.mean() / model.num_hidden_layers
        cost = (1.0 - acc) + args.cost_lambda * layer_fraction
        sweep.append(
            {
                "tau": float(tau),
                "accuracy": float(acc),
                "retention": float(retention),
                "layer_fraction": float(layer_fraction),
                "cost": float(cost),
            }
        )

    feasible = [
        s
        for s in sweep
        if s["retention"] >= args.target_retention
        and s["layer_fraction"] <= args.max_layer_fraction
    ]
    if feasible:
        best = sorted(
            feasible, key=lambda x: (-x["retention"], -x["accuracy"], x["layer_fraction"])
        )[0]
    else:
        best = sorted(sweep, key=lambda x: x["cost"])[0]
    tau = float(best["tau"])
    print(f"Selected tau={tau:.3f}")

    print("=== Test evaluation ===")
    t0 = time.time()
    test_cache = model.precompute_cache(test_texts, batch_size=args.batch_size)
    full_time = time.time() - t0

    teacher_preds = test_cache["final_preds"]
    teacher_acc = accuracy_score(test_labels, teacher_preds)
    teacher_f1 = f1_score(test_labels, teacher_preds, average="macro", zero_division=0)

    early_preds, used_layers = model.compose_predictions(test_cache, tau=tau)
    early_acc = accuracy_score(test_labels, early_preds)
    early_f1 = f1_score(test_labels, early_preds, average="macro", zero_division=0)
    retention = early_acc / teacher_acc if teacher_acc > 0 else 0.0
    avg_layers = float(used_layers.mean())
    layer_fraction = avg_layers / float(model.num_hidden_layers)

    # Approximate runtime scales with fraction of layers executed.
    approx_early_time = full_time * layer_fraction
    n = len(test_texts)
    results = {
        "version": "v3_early_exit_finbert",
        "config": vars(args),
        "model": {
            "num_hidden_layers": model.num_hidden_layers,
            "exit_layers": list(model.exit_layers),
        },
        "training_summary": train_summary,
        "threshold_selection": {
            "selected_tau": float(tau),
            "selected_point": best,
            "validation_teacher_acc": float(val_teacher_acc),
            "sweep": sweep,
        },
        "test_metrics": {
            "teacher_accuracy": float(teacher_acc),
            "teacher_f1": float(teacher_f1),
            "early_exit_accuracy": float(early_acc),
            "early_exit_f1": float(early_f1),
            "accuracy_retention": float(retention),
            "avg_layers_used": float(avg_layers),
            "layer_fraction": float(layer_fraction),
            "meets_retention_target": bool(retention >= args.target_retention),
            "meets_layer_budget": bool(layer_fraction <= args.max_layer_fraction),
        },
        "efficiency": {
            "full_finbert_time_s": float(full_time),
            "full_finbert_throughput": float(n / max(full_time, 1e-9)),
            "approx_early_exit_time_s": float(approx_early_time),
            "approx_early_exit_throughput": float(n / max(approx_early_time, 1e-9)),
            "approx_speedup_vs_full": float(1.0 / max(layer_fraction, 1e-9)),
        },
    }

    out_json = out_dir / "all_results_v3_early_exit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
