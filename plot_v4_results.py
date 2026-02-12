"""Generate plots from v4 quantized early-exit router results JSON."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_threshold_sweep(results: dict, out_dir: Path):
    """Accuracy and early-exit % vs tau threshold."""
    sweep = results["threshold_selection"]["sweep"]
    taus = [p["tau"] for p in sweep]
    accs = [p["accuracy"] for p in sweep]
    early_pcts = [p["early_exit_pct"] for p in sweep]
    sel_tau = results["threshold_selection"]["selected_tau"]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_acc = "steelblue"
    color_exit = "coral"

    ax1.plot(taus, accs, color=color_acc, linewidth=2, label="Hybrid Accuracy")
    ax1.set_xlabel("Router Threshold (τ)", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12, color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0.65, 0.90)

    ax2 = ax1.twinx()
    ax2.plot(taus, early_pcts, color=color_exit, linewidth=2, linestyle="--", label="Early Exit %")
    ax2.set_ylabel("Early Exit %", fontsize=12, color=color_exit)
    ax2.tick_params(axis="y", labelcolor=color_exit)
    ax2.set_ylim(0, 105)

    ax1.axvline(sel_tau, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.annotate(
        f"τ={sel_tau}",
        xy=(sel_tau, 0.67),
        fontsize=10,
        ha="center",
        color="gray",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=10)

    ax1.set_title("V4: Threshold Sweep — Accuracy vs Early Exit Rate", fontsize=13)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "v4_01_threshold_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved v4_01_threshold_sweep.png")


def plot_pareto_frontier(results: dict, out_dir: Path):
    """Accuracy retention vs layer fraction (cost)."""
    sweep = results["threshold_selection"]["sweep"]
    val_full_acc = results["threshold_selection"]["validation_full_accuracy"]
    layer_fracs = [p["layer_fraction"] for p in sweep]
    retentions = [p["accuracy"] / val_full_acc if val_full_acc > 0 else 0 for p in sweep]
    sel_tau = results["threshold_selection"]["selected_tau"]
    sel_point = results["threshold_selection"]["selected_point"]
    sel_ret = sel_point["accuracy"] / val_full_acc if val_full_acc > 0 else 0
    sel_frac = sel_point["layer_fraction"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(layer_fracs, retentions, c="steelblue", s=20, alpha=0.6, label="Sweep points")
    ax.scatter([sel_frac], [sel_ret], c="red", s=120, zorder=5, marker="*", label=f"Selected τ={sel_tau}")

    ax.axhline(0.98, color="green", linestyle="--", alpha=0.5, label="98% retention target")
    ax.axvline(0.35, color="orange", linestyle="--", alpha=0.5, label="35% layer budget")

    ax.set_xlabel("Layer Fraction (cost proxy)", fontsize=12)
    ax.set_ylabel("Accuracy Retention", fontsize=12)
    ax.set_title("V4: Pareto Frontier — Retention vs Computational Cost", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "v4_02_pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved v4_02_pareto_frontier.png")


def plot_efficiency_summary(results: dict, out_dir: Path):
    """Bar chart comparing full vs hybrid runtime and model size."""
    eff = results["efficiency"]
    mem = results["memory"]
    test = results["test_metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Runtime
    labels_rt = ["Full FinBERT\n(INT8)", "Hybrid\n(early-exit)"]
    times = [eff["full_quantized_finbert_time_s"], eff["hybrid_time_s"]]
    colors_rt = ["steelblue", "coral"]
    axes[0].bar(labels_rt, times, color=colors_rt, width=0.5)
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title(f"Test-set Inference Time\n(speedup: {eff['runtime_speedup_vs_full']:.2f}x)")
    for i, t in enumerate(times):
        axes[0].text(i, t + 0.3, f"{t:.1f}s", ha="center", fontsize=10)

    # Model size
    labels_sz = ["FP32", "INT8"]
    sizes = [mem["fp32_model_size_mb"], mem["int8_model_size_mb"]]
    colors_sz = ["steelblue", "seagreen"]
    axes[1].bar(labels_sz, sizes, color=colors_sz, width=0.5)
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title(f"Model Size\n(reduction: {mem['int8_size_reduction_pct']:.1f}%)")
    for i, s in enumerate(sizes):
        axes[1].text(i, s + 5, f"{s:.0f}MB", ha="center", fontsize=10)

    # Accuracy comparison
    labels_acc = ["Full FinBERT", "Hybrid"]
    accs = [test["full_accuracy"], test["hybrid_accuracy"]]
    colors_acc = ["steelblue", "coral"]
    axes[2].bar(labels_acc, accs, color=colors_acc, width=0.5)
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title(f"Test Accuracy\n(retention: {test['accuracy_retention']:.1%})")
    axes[2].set_ylim(0, 1.0)
    for i, a in enumerate(accs):
        axes[2].text(i, a + 0.01, f"{a:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "v4_03_efficiency_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved v4_03_efficiency_summary.png")


def plot_layer_usage(results: dict, out_dir: Path):
    """Average layers used and stage time breakdown."""
    eff = results["efficiency"]
    test = results["test_metrics"]
    stage_times = eff["stage_times_s"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Layer usage pie
    early_pct = test["early_exit_pct"]
    deferred_pct = test["deferred_pct"]
    axes[0].pie(
        [early_pct, deferred_pct],
        labels=[f"Early Exit (L1-4)\n{early_pct:.1f}%", f"Full (L1-12)\n{deferred_pct:.1f}%"],
        colors=["seagreen", "steelblue"],
        autopct="",
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[0].set_title(f"Sample Routing\n(avg layers: {test['avg_layers_used']:.1f}/12)", fontsize=12)

    # Stage time breakdown
    stage_labels = ["Stage 1\n(Layers 1-4)", "Router", "Stage 2\n(Layers 5-12)"]
    stage_vals = [stage_times["stage1_layers_1_4"], stage_times["router"], stage_times["stage2_layers_5_12"]]
    colors_stage = ["seagreen", "gold", "steelblue"]
    bars = axes[1].barh(stage_labels, stage_vals, color=colors_stage, height=0.5)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_title("Staged Execution Time Breakdown", fontsize=12)
    for bar, v in zip(bars, stage_vals):
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{v:.2f}s", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "v4_04_layer_usage.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved v4_04_layer_usage.png")


def plot_router_feature_importance(results: dict, out_dir: Path):
    """Feature importance of the XGBoost router."""
    fi = results["training"]["router"]["feature_importance"]
    names = list(fi.keys())
    values = list(fi.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, values, color="steelblue", height=0.5)
    ax.set_xlabel("Feature Importance")
    ax.set_title("V4: XGBoost Router Feature Importance", fontsize=13)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig(out_dir / "v4_05_router_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved v4_05_router_feature_importance.png")


def plot_cost_vs_accuracy(results: dict, out_dir: Path):
    """Cost metric vs accuracy across the sweep."""
    sweep = results["threshold_selection"]["sweep"]
    costs = [p["cost"] for p in sweep]
    accs = [p["accuracy"] for p in sweep]
    taus = [p["tau"] for p in sweep]
    sel_tau = results["threshold_selection"]["selected_tau"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(accs, costs, c=taus, cmap="viridis", s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Threshold τ")

    sel_point = results["threshold_selection"]["selected_point"]
    ax.scatter([sel_point["accuracy"]], [sel_point["cost"]], c="red", s=150, marker="*", zorder=5, label=f"Selected τ={sel_tau}")

    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel("Cost (combined metric)", fontsize=12)
    ax.set_title("V4: Cost vs Accuracy Trade-off", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "v4_06_cost_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved v4_06_cost_vs_accuracy.png")


def main():
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "prev_results/results_v4_full/all_results_v4_quantized_early_exit_router.json"

    if len(sys.argv) > 2:
        out_dir = Path(sys.argv[2])
    else:
        out_dir = Path(json_path).parent

    out_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(json_path)

    print(f"Loaded results from {json_path}")
    print(f"Saving plots to {out_dir}/\n")

    plot_threshold_sweep(results, out_dir)
    plot_pareto_frontier(results, out_dir)
    plot_efficiency_summary(results, out_dir)
    plot_layer_usage(results, out_dir)
    plot_router_feature_importance(results, out_dir)
    plot_cost_vs_accuracy(results, out_dir)

    # Print summary
    test = results["test_metrics"]
    eff = results["efficiency"]
    mem = results["memory"]
    print("\n=== V4 Results Summary ===")
    print(f"Full FinBERT (INT8): {test['full_accuracy']:.4f} acc, {test['full_f1']:.4f} F1")
    print(f"Hybrid:              {test['hybrid_accuracy']:.4f} acc, {test['hybrid_f1']:.4f} F1")
    print(f"Retention:           {test['accuracy_retention']:.2%}")
    print(f"Early exit rate:     {test['early_exit_pct']:.1f}%")
    print(f"Avg layers:          {test['avg_layers_used']:.1f}/12 ({test['layer_fraction']:.1%})")
    print(f"Speedup (measured):  {eff['runtime_speedup_vs_full']:.2f}x")
    print(f"INT8 size reduction: {mem['int8_size_reduction_pct']:.1f}%")
    print(f"McNemar p-value:     {test['mcnemar_p_value']:.2e}")
    print(f"Retention CI (95%):  [{test['ci_retention_95'][0]:.4f}, {test['ci_retention_95'][1]:.4f}]")
    print(f"Meets retention target (98%): {test['meets_retention_target']}")
    print(f"Meets layer budget (35%):     {test['meets_layer_budget']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
