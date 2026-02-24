"""
P6: Unified Agreement-Tier Analysis Figure

Creates a comprehensive figure showing how annotator agreement correlates
with routing decisions across all four experiments. This is the paper's
strongest finding — the key secondary contribution.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    out_dir = Path("results_agreement_analysis")
    out_dir.mkdir(exist_ok=True)

    tiers = ["100", "75", "66", "50"]
    tier_labels = ["AllAgree", "75%", "66%", "50%"]
    tier_numeric = [1.0, 0.75, 0.66, 0.50]  # for correlation

    # ============================================================
    # Load results from all experiments
    # ============================================================
    experiments = {}

    # V1: VADER Cascade
    v1_path = Path("all_results/results/all_results.json")
    if v1_path.exists():
        v1 = load_json(v1_path)
        if "agreement_routing" in v1:
            ar = v1["agreement_routing"]
            experiments["Exp A: VADER Cascade"] = {
                "full_acc": {t: ar.get(t, {}).get("finbert_acc", 0) for t in tiers},
                "hybrid_acc": {t: ar.get(t, {}).get("hybrid_acc", 0) for t in tiers},
                "routing_metric": {t: ar.get(t, {}).get("finbert_usage_pct", 0) for t in tiers},
                "routing_label": "FinBERT Usage (%)",
            }

    # V2: KD Student
    v2_path = Path("all_results/results_v2_full_kd/all_results_v2.json")
    if v2_path.exists():
        v2 = load_json(v2_path)
        if "expD_agreement_tier" in v2:
            at = v2["expD_agreement_tier"]
            experiments["Exp B: KD Student"] = {
                "full_acc": {t: at.get(t, {}).get("teacher_acc", 0) for t in tiers},
                "hybrid_acc": {t: at.get(t, {}).get("hybrid_acc", 0) for t in tiers},
                "routing_metric": {t: at.get(t, {}).get("teacher_pct", 0) for t in tiers},
                "routing_label": "Teacher Usage (%)",
            }

    # V5: Multi-Exit PABEE
    v5_path = Path("all_results/results_v5_full/all_results_v5_multi_exit.json")
    if v5_path.exists():
        v5 = load_json(v5_path)
        if "agreement_tier_analysis" in v5:
            at = v5["agreement_tier_analysis"]
            experiments["Exp D: Multi-Exit"] = {
                "full_acc": {t: at.get(t, {}).get("full_acc", 0) for t in tiers},
                "hybrid_acc": {t: at.get(t, {}).get("hybrid_acc", 0) for t in tiers},
                "routing_metric": {t: at.get(t, {}).get("early_exit_pct", 0) for t in tiers},
                "routing_label": "Early Exit (%)",
            }

    # DeBERTa results (if available)
    deb_kd = Path("results_deberta/kd/deberta_kd_results.json")
    if deb_kd.exists():
        dk = load_json(deb_kd)
        if "agreement_tier_analysis" in dk:
            at = dk["agreement_tier_analysis"]
            experiments["DeBERTa KD"] = {
                "full_acc": {t: at.get(t, {}).get("teacher_acc", 0) for t in tiers},
                "hybrid_acc": {t: at.get(t, {}).get("hybrid_acc", 0) for t in tiers},
                "routing_metric": {t: at.get(t, {}).get("teacher_pct", 0) for t in tiers},
                "routing_label": "Teacher Usage (%)",
            }

    deb_me = Path("results_deberta/multi_exit/deberta_multi_exit_results.json")
    if deb_me.exists():
        dm = load_json(deb_me)
        if "agreement_tier_analysis" in dm:
            at = dm["agreement_tier_analysis"]
            experiments["DeBERTa Multi-Exit"] = {
                "full_acc": {t: at.get(t, {}).get("full_acc", 0) for t in tiers},
                "hybrid_acc": {t: at.get(t, {}).get("hybrid_acc", 0) for t in tiers},
                "routing_metric": {t: at.get(t, {}).get("early_exit_pct", 0) for t in tiers},
                "routing_label": "Early Exit (%)",
            }

    # NeoBERT results
    neo_path = Path("results_neobert/all_neobert_results.json")
    if neo_path.exists():
        neo = load_json(neo_path)
        # NeoBERT KD
        if "kd_routing" in neo and "agreement_tier_analysis" in neo["kd_routing"]:
            at = neo["kd_routing"]["agreement_tier_analysis"]
            experiments["NeoBERT KD"] = {
                "full_acc": {t: at.get(t, {}).get("teacher_acc", 0) for t in tiers},
                "hybrid_acc": {t: at.get(t, {}).get("hybrid_acc", 0) for t in tiers},
                "routing_metric": {t: at.get(t, {}).get("teacher_pct", 0) for t in tiers},
                "routing_label": "Teacher Usage (%)",
            }
        # NeoBERT Multi-Exit
        if "multi_exit" in neo and "agreement_tier_analysis" in neo["multi_exit"]:
            at = neo["multi_exit"]["agreement_tier_analysis"]
            experiments["NeoBERT Multi-Exit"] = {
                "full_acc": {t: at.get(t, {}).get("full_acc", 0) for t in tiers},
                "hybrid_acc": {t: at.get(t, {}).get("hybrid_acc", 0) for t in tiers},
                "routing_metric": {t: at.get(t, {}).get("early_exit_pct", 0) for t in tiers},
                "routing_label": "Early Exit (%)",
            }

    n_exp = len(experiments)
    if n_exp == 0:
        print("No experiment results found!")
        return

    print(f"Found {n_exp} experiments: {list(experiments.keys())}")

    # ============================================================
    # Figure 1: 4-panel (2 rows: accuracy + routing metric per tier)
    # ============================================================
    fig, axes = plt.subplots(2, n_exp, figsize=(5 * n_exp, 8), squeeze=False)

    colors_full = "steelblue"
    colors_hybrid = "coral"

    for idx, (name, data) in enumerate(experiments.items()):
        x = np.arange(len(tiers))

        # Top row: accuracy per tier
        ax = axes[0, idx]
        full_accs = [data["full_acc"].get(t, 0) for t in tiers]
        hybrid_accs = [data["hybrid_acc"].get(t, 0) for t in tiers]
        w = 0.35
        ax.bar(x - w/2, full_accs, w, label="Full Model", color=colors_full, alpha=0.8)
        ax.bar(x + w/2, hybrid_accs, w, label="Hybrid", color=colors_hybrid, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

        # Bottom row: routing metric per tier
        ax2 = axes[1, idx]
        routing = [data["routing_metric"].get(t, 0) for t in tiers]
        bars = ax2.bar(x, routing, color="mediumpurple", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(tier_labels)
        ax2.set_xlabel("Agreement Tier")
        ax2.set_ylabel(data["routing_label"])

        # Add value labels
        for bar, val in zip(bars, routing):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Agreement-Tier Routing Patterns Across Experiments",
                fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "unified_agreement_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved unified analysis to {out_dir / 'unified_agreement_analysis.png'}")

    # ============================================================
    # Figure 2: Spearman correlations
    # ============================================================
    print("\n=== Spearman Correlations: Agreement Tier vs Routing ===")
    correlations = {}
    for name, data in experiments.items():
        routing_vals = [data["routing_metric"].get(t, 0) for t in tiers]
        acc_drops = [
            data["full_acc"].get(t, 0) - data["hybrid_acc"].get(t, 0)
            for t in tiers
        ]

        if len(set(routing_vals)) > 1:
            rho_routing, p_routing = spearmanr(tier_numeric, routing_vals)
        else:
            rho_routing, p_routing = 0.0, 1.0
        if len(set(acc_drops)) > 1:
            rho_drop, p_drop = spearmanr(tier_numeric, acc_drops)
        else:
            rho_drop, p_drop = 0.0, 1.0

        correlations[name] = {
            "routing_rho": float(rho_routing),
            "routing_p": float(p_routing),
            "acc_drop_rho": float(rho_drop),
            "acc_drop_p": float(p_drop),
        }
        print(f"  {name}:")
        print(f"    Tier vs Routing: rho={rho_routing:.3f}, p={p_routing:.3f}")
        print(f"    Tier vs AccDrop: rho={rho_drop:.3f}, p={p_drop:.3f}")

    # Correlation bar chart
    if correlations:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(correlations.keys())
        x = np.arange(len(names))
        w = 0.35
        rho_routing = [correlations[n]["routing_rho"] for n in names]
        rho_drop = [correlations[n]["acc_drop_rho"] for n in names]

        ax.bar(x - w/2, rho_routing, w, label="Tier vs Routing Rate", color="steelblue", alpha=0.8)
        ax.bar(x + w/2, rho_drop, w, label="Tier vs Accuracy Drop", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Agreement Tier Correlations with Routing Behavior")
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / "agreement_correlations.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved correlations plot to {out_dir / 'agreement_correlations.png'}")

    # Save results
    summary = {
        "experiments": {
            name: {
                "tiers": tiers,
                "full_acc": data["full_acc"],
                "hybrid_acc": data["hybrid_acc"],
                "routing_metric": data["routing_metric"],
                "routing_label": data["routing_label"],
            }
            for name, data in experiments.items()
        },
        "correlations": correlations,
    }
    with open(out_dir / "agreement_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved analysis to {out_dir / 'agreement_analysis.json'}")


if __name__ == "__main__":
    main()
