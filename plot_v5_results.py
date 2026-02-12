"""Generate plots for v5 multi-exit experiment results."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LABEL_LIST = ["negative", "neutral", "positive"]


def load_results(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def plot_threshold_sweep(results: dict, out_dir: Path):
    """Plot 1: Threshold sweep — accuracy & retention vs tau."""
    sweep = results["threshold_selection"]["sweep"]
    taus = [s["tau"] for s in sweep]
    accs = [s["accuracy"] for s in sweep]
    retentions = [s["retention"] for s in sweep]
    early_pcts = [s["early_exit_pct"] for s in sweep]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(taus, accs, "b-", linewidth=2, label="Accuracy")
    ax1.plot(taus, retentions, "b--", linewidth=2, alpha=0.7, label="Retention")
    ax1.set_xlabel("Router Confidence Threshold (τ)")
    ax1.set_ylabel("Performance", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0.5, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(taus, early_pcts, "r-", linewidth=2, label="Early Exit %")
    ax2.set_ylabel("Early Exit %", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Mark selected tau
    sel_tau = results["threshold_selection"]["selected_tau"]
    ax1.axvline(x=sel_tau, color="gray", linestyle=":", alpha=0.7, label=f"τ={sel_tau:.3f}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    ax1.set_title("Multi-Exit Threshold Sensitivity")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "01_v5_threshold_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pareto_frontier(results: dict, out_dir: Path):
    """Plot 2: Pareto frontier — accuracy vs computational cost (layer fraction)."""
    sweep = results["threshold_selection"]["sweep"]
    lf = [s["layer_fraction"] for s in sweep]
    accs = [s["accuracy"] for s in sweep]

    # Also plot entropy baseline
    ent_sweep = results["entropy_baseline"]["sweep"]
    ent_lf = [s["layer_fraction"] for s in ent_sweep]
    ent_accs = [s["accuracy"] for s in ent_sweep]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lf, accs, "o-", color="steelblue", markersize=2, label="Learned Router", alpha=0.8)
    ax.plot(ent_lf, ent_accs, "s-", color="coral", markersize=2, label="Entropy Baseline", alpha=0.6)

    # Mark full FinBERT
    full_acc = results["test_metrics"]["full_accuracy"]
    ax.axhline(y=full_acc, color="gray", linestyle="--", alpha=0.5, label=f"Full FinBERT ({full_acc:.3f})")

    # Mark our operating point
    sel = results["threshold_selection"]["selected_point"]
    ax.plot(sel["layer_fraction"], sel["accuracy"], "*", color="gold", markersize=15,
            markeredgecolor="black", label=f"Selected (acc={sel['accuracy']:.3f})", zorder=5)

    ax.set_xlabel("Layer Fraction (computational cost)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontier: Learned Router vs Entropy Baseline")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "02_v5_pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_layer_usage(results: dict, out_dir: Path):
    """Plot 3: Layer usage distribution (pie/bar)."""
    usage = results["test_metrics"]["layer_usage_distribution"]
    layers = list(usage.keys())
    pcts = list(usage.values())
    colors = ["#4CAF50", "#2196F3", "#FF5722"][:len(layers)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar(layers, pcts, color=colors)
    axes[0].set_xlabel("Exit Layer")
    axes[0].set_ylabel("Samples (%)")
    axes[0].set_title("Layer Usage Distribution")
    for i, (l, p) in enumerate(zip(layers, pcts)):
        axes[0].text(i, p + 1, f"{p:.1f}%", ha="center", fontsize=10)

    # Pie chart
    labels = [f"Layer {l}\n({p:.1f}%)" for l, p in zip(layers, pcts)]
    axes[1].pie(pcts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Exit Distribution")

    plt.tight_layout()
    plt.savefig(out_dir / "03_v5_layer_usage.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results: dict, out_dir: Path):
    """Plot 4: Model comparison bar chart (TF-IDF vs exit heads vs hybrid vs full)."""
    models = ["TF-IDF", "Exit L4", "Exit L8", "Hybrid", "Full FinBERT"]
    accs = [
        results["tfidf_baseline"]["accuracy"],
        results["exit_layer_metrics"]["4"]["accuracy"],
        results["exit_layer_metrics"]["8"]["accuracy"],
        results["test_metrics"]["hybrid_accuracy"],
        results["test_metrics"]["full_accuracy"],
    ]
    f1s = [
        results["tfidf_baseline"]["f1_macro"],
        results["exit_layer_metrics"]["4"]["f1_macro"],
        results["exit_layer_metrics"]["8"]["f1_macro"],
        results["test_metrics"]["hybrid_f1"],
        results["test_metrics"]["full_f1"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + w/2, f1s, w, label="F1-Macro", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "04_v5_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(results: dict, out_dir: Path):
    """Plot 5: Confusion matrices for Full FinBERT, Hybrid, and TF-IDF."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, cm_key) in zip(axes, [
        ("Full FinBERT", results["test_metrics"]["full_confusion_matrix"]),
        ("Hybrid Multi-Exit", results["test_metrics"]["hybrid_confusion_matrix"]),
        ("TF-IDF Baseline", results["tfidf_baseline"]["confusion_matrix"]),
    ]):
        cm = np.array(cm_key)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_LIST, yticklabels=LABEL_LIST, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}")

    plt.tight_layout()
    plt.savefig(out_dir / "05_v5_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_agreement_tier(results: dict, out_dir: Path):
    """Plot 6: Performance by agreement tier."""
    tier_data = results["agreement_tier_analysis"]
    tiers = sorted(tier_data.keys(), key=lambda x: int(x), reverse=True)
    tier_labels = [f"{t}%" for t in tiers]

    full_accs = [tier_data[t]["full_acc"] for t in tiers]
    hybrid_accs = [tier_data[t]["hybrid_acc"] for t in tiers]
    tfidf_accs = [tier_data[t]["tfidf_acc"] for t in tiers]
    exit_pcts = [tier_data[t]["early_exit_pct"] for t in tiers]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(tiers))
    w = 0.25

    axes[0].bar(x - w, full_accs, w, label="Full FinBERT", color="coral")
    axes[0].bar(x, hybrid_accs, w, label="Hybrid", color="steelblue")
    axes[0].bar(x + w, tfidf_accs, w, label="TF-IDF", color="seagreen")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tier_labels)
    axes[0].set_xlabel("Agreement Tier")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Per-Tier Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(tier_labels, exit_pcts, color="steelblue")
    axes[1].set_xlabel("Agreement Tier")
    axes[1].set_ylabel("Early Exit %")
    axes[1].set_title("Per-Tier Early Exit Rate")

    plt.tight_layout()
    plt.savefig(out_dir / "06_v5_agreement_tier.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_efficiency(results: dict, out_dir: Path):
    """Plot 7: Efficiency comparison (throughput + speedup)."""
    eff = results["efficiency"]
    models = ["Full FinBERT", "Hybrid Multi-Exit", "TF-IDF"]
    throughputs = [
        eff["full_finbert_throughput"],
        eff["hybrid_throughput"],
        eff["tfidf_throughput"],
    ]
    times = [
        eff["full_finbert_time_s"],
        eff["hybrid_time_s"],
        eff["tfidf_time_s"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["coral", "steelblue", "seagreen"]

    bars = axes[0].bar(models, throughputs, color=colors)
    axes[0].set_ylabel("Samples / sec")
    axes[0].set_title("Throughput (GPU)")
    axes[0].set_yscale("log")
    for bar, t in zip(bars, throughputs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                     f"{t:.0f}", ha="center", fontsize=9)

    bars = axes[1].bar(models, times, color=colors)
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_title("Total Inference Time")
    for bar, t in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width()/2, t + max(times)*0.02,
                     f"{t:.3f}s", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "07_v5_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_router_importance(results: dict, out_dir: Path):
    """Plot 8: Router feature importance."""
    importance = results["training"]["router"]["feature_importance"]
    if not importance or all(v == 0 for v in importance.values()):
        return

    names = list(importance.keys())
    values = list(importance.values())
    sorted_idx = np.argsort(values)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in sorted_idx], [values[i] for i in sorted_idx], color="steelblue")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Router Feature Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "08_v5_router_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "results_v5_full/all_results_v5_multi_exit.json"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(json_path).parent

    results = load_results(json_path)
    out_dir.mkdir(exist_ok=True)

    plot_threshold_sweep(results, out_dir)
    plot_pareto_frontier(results, out_dir)
    plot_layer_usage(results, out_dir)
    plot_model_comparison(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_agreement_tier(results, out_dir)
    plot_efficiency(results, out_dir)
    plot_router_importance(results, out_dir)

    print(f"Generated 8 plots in {out_dir}")

    # Print summary table
    tm = results["test_metrics"]
    eff = results["efficiency"]
    tfidf = results["tfidf_baseline"]
    print(f"\n{'Model':<20} {'Acc':>8} {'F1':>8} {'Throughput':>12} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'TF-IDF':<20} {tfidf['accuracy']:>8.4f} {tfidf['f1_macro']:>8.4f} "
          f"{eff['tfidf_throughput']:>10.1f}/s {'--':>10}")
    for layer, m in results["exit_layer_metrics"].items():
        print(f"{'Exit Layer ' + layer:<20} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f} "
              f"{'--':>12} {'--':>10}")
    print(f"{'Hybrid Multi-Exit':<20} {tm['hybrid_accuracy']:>8.4f} {tm['hybrid_f1']:>8.4f} "
          f"{eff['hybrid_throughput']:>10.1f}/s {eff['runtime_speedup_vs_full']:>9.2f}x")
    print(f"{'Full FinBERT':<20} {tm['full_accuracy']:>8.4f} {tm['full_f1']:>8.4f} "
          f"{eff['full_finbert_throughput']:>10.1f}/s {'1.00x':>10}")


if __name__ == "__main__":
    main()
