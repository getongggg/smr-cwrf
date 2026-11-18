from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "closed_loop" / "test" / "summary.csv"
PAIRED = ROOT / "results" / "closed_loop" / "test" / "paired_statistics.csv"
GAP = ROOT / "results" / "smr_mechanism" / "oracle_gap_decomposition.csv"
OUTPUT = ROOT / "manuscript" / "figures" / "figure3_closed_loop_mechanism"
SOURCE = ROOT / "results" / "paper_tables" / "figure3_closed_loop_mechanism_source.csv"


COLORS = {
    "cwrf": "#6B7280",
    "union": "#D6A84B",
    "smr_cwrf": "#287D8E",
    "headroom": "#B8C2CC",
    "captured": "#287D8E",
    "residual": "#D6A84B",
}


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=8)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })
    summary = pd.read_csv(SUMMARY).set_index("method")
    paired = pd.read_csv(PAIRED)
    gap = pd.read_csv(GAP)

    fig = plt.figure(figsize=(7.15, 2.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.15, 1.0))

    ax = fig.add_subplot(grid[0, 0])
    metrics = ("collision_rate", "violation_rate", "success_rate")
    labels = ("Collision", "Safety violation", "Completion")
    methods = ("cwrf", "union", "smr_cwrf")
    y = np.arange(len(metrics))
    offsets = (-0.18, 0.0, 0.18)
    for method, offset in zip(methods, offsets):
        values = 100.0 * summary.loc[method, list(metrics)].to_numpy(float)
        ax.scatter(values, y + offset, s=24, color=COLORS[method], label={"cwrf": "CWRF", "union": "Union", "smr_cwrf": "SMR-CWRF"}[method], zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Closed-loop episode rate (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6)
    ax.set_title("1,000 independent test episodes", fontsize=7.2, pad=16)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    rows = paired[(paired["candidate"] == "smr_cwrf")].set_index("metric")
    effect_metrics = ("collision", "violation", "success", "cumulative_exposure", "path_length_ratio", "replans")
    effect_labels = ("Collision", "Violation", "Completion", "Exposure", "Length ratio", "Replans")
    baseline_values = {
        "collision": summary.loc["cwrf", "collision_rate"],
        "violation": summary.loc["cwrf", "violation_rate"],
        "success": summary.loc["cwrf", "success_rate"],
        "cumulative_exposure": summary.loc["cwrf", "exposure_mean"],
        "path_length_ratio": summary.loc["cwrf", "length_ratio_mean"],
        "replans": summary.loc["cwrf", "replans_mean"],
    }
    directions = {metric: (-1.0 if metric != "success" else 1.0) for metric in effect_metrics}
    points, lows, highs = [], [], []
    for metric in effect_metrics:
        row = rows.loc[metric]
        scale = 100.0 / float(baseline_values[metric])
        direction = directions[metric]
        transformed = direction * np.asarray([
            row["candidate_minus_baseline_mean"], row["bootstrap_95_low"], row["bootstrap_95_high"]
        ]) * scale
        points.append(transformed[0])
        lows.append(min(transformed[1:]))
        highs.append(max(transformed[1:]))
    yy = np.arange(len(effect_metrics))
    ax.axvline(0, color="#374151", linewidth=0.8)
    ax.errorbar(
        points,
        yy,
        xerr=[np.asarray(points) - np.asarray(lows), np.asarray(highs) - np.asarray(points)],
        fmt="o",
        color=COLORS["smr_cwrf"],
        ecolor=COLORS["smr_cwrf"],
        markersize=4.5,
        capsize=2,
        linewidth=1.0,
    )
    ax.set_yticks(yy, effect_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative improvement vs CWRF (%)")
    ax.set_title("Paired effect; positive is favorable", fontsize=7.2)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[0, 2])
    gap_metrics = ("violation", "risk_exposure", "minimum_clearance")
    gap_labels = ("Violation", "Exposure", "Clearance")
    x = np.arange(len(gap_metrics))
    captured_share = []
    residual_share = []
    for metric in gap_metrics:
        current = gap[gap["metric"] == metric].set_index("component")
        headroom = float(current.loc["candidate_headroom_cwrf_minus_oracle", "improvement_positive_mean"])
        captured = float(current.loc["captured_by_smr_cwrf_minus_smr", "improvement_positive_mean"])
        captured_share.append(100.0 * captured / headroom)
        residual_share.append(100.0 * (headroom - captured) / headroom)
    ax.bar(x, captured_share, color=COLORS["captured"], width=0.58)
    ax.bar(x, residual_share, bottom=captured_share, color=COLORS["residual"], width=0.58)
    ax.set_xticks(x, gap_labels, rotation=18, ha="right")
    ax.set_ylabel("Candidate headroom decomposition (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Ground-truth oracle diagnostic", fontsize=7.2)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.text(0, max(captured_share[0] / 2, 2.0), "captured", color="white", ha="center", va="center", fontsize=5.8, rotation=90)
    ax.text(0, captured_share[0] + residual_share[0] / 2, "residual selector gap", color="#374151", ha="center", va="center", fontsize=5.8, rotation=90)
    ax.text(0.98, 0.03, "SMR-oracle match: 48.5%", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#374151")
    panel_label(ax, "c")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for method in methods:
        for metric in metrics:
            source_rows.append({"panel": "a", "method": method, "metric": metric, "value": summary.loc[method, metric]})
    for metric, point, low, high in zip(effect_metrics, points, lows, highs):
        source_rows.append({"panel": "b", "method": "smr_cwrf", "metric": metric, "value": point, "ci_low": low, "ci_high": high})
    for metric, captured, residual in zip(gap_metrics, captured_share, residual_share):
        source_rows.append({"panel": "c", "method": "smr_cwrf", "metric": metric, "captured_percent": captured, "residual_percent": residual})
    pd.DataFrame(source_rows).to_csv(SOURCE, index=False)

    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
