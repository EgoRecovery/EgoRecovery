#!/usr/bin/env python3
"""Render the additional rebuttal tables as reviewer-readable figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 11,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "rebuttal_figures"

INK = "#1F2937"
MUTED = "#667085"
GRID = "#D8DEE9"
BG = "#FFFFFF"
BASELINE_DARK = "#4F628E"
BASELINE_MID = "#8193BC"
BASELINE_SOFT = "#C7D1E7"
OURS = "#C96F88"
OURS_DARK = "#9F4964"
OURS_SOFT = "#F4DEE5"
BLUE_SCALE = LinearSegmentedColormap.from_list(
    "egorecovery_blue", ["#F5F7FB", "#DCE5F3", "#9EB2D4", "#4F6FA8"]
)


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(colors=INK, labelsize=10)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor=BG)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor=BG)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def annotated_heatmap(
    stem: str,
    row_labels: list[str],
    col_labels: list[str],
    values: np.ndarray,
    highlight_row: int,
) -> None:
    height = max(5.4, 2.8 + 0.82 * len(row_labels))
    fig, ax = plt.subplots(figsize=(14.5, height))
    fig.patch.set_facecolor(BG)

    im = ax.imshow(values, cmap=BLUE_SCALE, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    for label in ax.get_xticklabels():
        label.set_fontsize(15)
        label.set_fontweight("semibold")
    for label in ax.get_yticklabels():
        label.set_fontsize(15)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text_color = "white" if value >= 72 else INK
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", color=text_color, fontsize=20, fontweight="bold")

    for x in np.arange(-0.5, values.shape[1], 1):
        ax.axvline(x, color="white", linewidth=2)
    for y in np.arange(-0.5, values.shape[0], 1):
        ax.axhline(y, color="white", linewidth=2)

    ax.add_patch(
        Rectangle(
            (-0.54, highlight_row - 0.43),
            0.035,
            0.86,
            facecolor=OURS,
            edgecolor="none",
            clip_on=False,
            zorder=5,
        )
    )
    ax.get_yticklabels()[highlight_row].set_color(OURS_DARK)
    ax.get_yticklabels()[highlight_row].set_fontweight("bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("Success rate (%)", color=MUTED, fontsize=13)
    cbar.ax.tick_params(labelsize=12, colors=MUTED, length=2)
    cbar.outline.set_visible(False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.38, right=0.93, top=0.86, bottom=0.08)
    save_figure(fig, stem)


def plot_t_a() -> None:
    annotated_heatmap(
        "table_ta_intent_representation",
        [
            "DCT-4D (ours)",
            "Raw future EE trajectory",
            "Future-trajectory latent",
            "Current/future observation latent",
        ],
        ["Cup-brush\nInitial", "Cup-brush\nRecovery", "Round-disk\nInitial", "Round-disk\nRecovery"],
        np.array([[75, 80, 85, 85], [70, 65, 75, 70], [65, 65, 75, 70], [55, 50, 60, 60]]),
        highlight_row=0,
    )


def plot_table_2_ablation() -> None:
    labels = [
        "Full EgoRecovery",
        "Direct human recovery mix",
        "without intent loss",
        "without corrective segment mask",
        "without intent modulation",
        r"$c_t$ zero at test time",
        "always-on modulation",
    ]
    initial = [80.0, 71.2, 78.7, 75.0, 76.3, 80.0, 70.0]
    recovery = [85.0, 71.2, 65.0, 55.0, 65.0, 65.0, 70.0]

    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    fig.patch.set_facecolor(BG)
    y = np.arange(len(labels))
    bar_h = 0.32
    ax.axhspan(-0.52, 0.52, color=OURS_SOFT, alpha=0.55, zorder=0)
    bars_initial = ax.barh(y - bar_h / 1.7, initial, height=bar_h, color=BASELINE_SOFT, edgecolor="none", label="Initial SR")
    bars_recovery = ax.barh(y + bar_h / 1.7, recovery, height=bar_h, color=BASELINE_DARK, edgecolor="none", label="Recovery SR")
    for bars, values in ((bars_initial, initial), (bars_recovery, recovery)):
        for bar, value in zip(bars, values):
            ax.text(value + 1.1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", ha="left", va="center", color=INK, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Success rate (%)", fontsize=15)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.65)
    style_axis(ax)
    ax.tick_params(labelsize=14)
    ax.get_yticklabels()[0].set_color(OURS_DARK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=legend_labels, loc="upper center", bbox_to_anchor=(0.62, 1.01), ncol=2, fontsize=13)
    fig.subplots_adjust(left=0.34, right=0.95, top=0.89, bottom=0.14)
    save_figure(fig, "table_2_mechanism_ablation")


def plot_table_6_recovery_supervision() -> None:
    labels = [
        "Robot success only",
        r"$R_{suc}(50) + H_{suc}(300)$",
        r"$R_{suc}(50) + H_{rec}(300)$",
        r"$R_{suc}(50) + R_{rec}(50)$",
        "+ Human success mix",
        "Direct human recovery mix",
        "EgoRecovery",
    ]
    initial = [37.5, 45.0, 42.5, 56.3, 63.8, 71.2, 80.0]
    recovery = [0.0, 0.0, 8.8, 52.5, 65.0, 71.2, 85.0]

    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    fig.patch.set_facecolor(BG)
    y = np.arange(len(labels))
    bar_h = 0.32
    ax.axhspan(5.48, 6.52, color=OURS_SOFT, alpha=0.55, zorder=0)
    bars_initial = ax.barh(y - bar_h / 1.7, initial, height=bar_h, color=BASELINE_SOFT, edgecolor="none", label="Initial SR")
    bars_recovery = ax.barh(y + bar_h / 1.7, recovery, height=bar_h, color=BASELINE_DARK, edgecolor="none", label="Recovery SR")
    for bars, values in ((bars_initial, initial), (bars_recovery, recovery)):
        for bar, value in zip(bars, values):
            x_text = value + 1.1
            ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", ha="left", va="center", color=INK, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Success rate (%)", fontsize=15)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.65)
    style_axis(ax)
    ax.tick_params(labelsize=14)
    ax.get_yticklabels()[-1].set_color(OURS_DARK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=legend_labels, loc="upper center", bbox_to_anchor=(0.62, 1.01), ncol=2, fontsize=13)
    fig.subplots_adjust(left=0.35, right=0.95, top=0.89, bottom=0.14)
    save_figure(fig, "table_6_recovery_supervision")


def plot_t_b() -> None:
    annotated_heatmap(
        "table_tb_gate_causality",
        [
            r"Learned $p_t$ (ours)",
            "$p_t = 0$\nnever inject",
            "$p_t = 0.5$\nconstant",
            "$p_t = 1$\nalways inject",
            r"Shuffled $p_t$",
        ],
        ["Cup-brush\nInitial", "Cup-brush\nRecovery", "Round-disk\nInitial", "Round-disk\nRecovery"],
        np.array([[75, 80, 85, 85], [65, 50, 75, 60], [40, 45, 50, 45], [35, 40, 40, 45], [30, 35, 40, 35]]),
        highlight_row=0,
    )


def plot_t_c() -> None:
    labels = ["DP, success only", "DP + robot recovery", "DP + EgoRecovery pathway"]
    cup_init = [35, 40, 60]
    cup_rec = [0, 45, 70]
    disk_init = [40, 50, 70]
    disk_rec = [0, 50, 65]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), sharex=True)
    fig.patch.set_facecolor(BG)
    for ax, task, initial, recovery in zip(
        axes,
        ["Cup-brush", "Round-disk"],
        [cup_init, disk_init],
        [cup_rec, disk_rec],
    ):
        y = np.arange(len(labels))
        bar_h = 0.25
        ax.axhspan(1.52, 2.48, color=OURS_SOFT, alpha=0.42, zorder=0)
        bars_init = ax.barh(
            y - bar_h / 1.7,
            initial,
            height=bar_h,
            color=BASELINE_SOFT,
            edgecolor="none",
            label="Initial SR",
        )
        bars_rec = ax.barh(
            y + bar_h / 1.7,
            recovery,
            height=bar_h,
            color=BASELINE_DARK,
            edgecolor="none",
            label="Recovery SR",
        )
        for bars, values in ((bars_init, initial), (bars_rec, recovery)):
            for bar, value in zip(bars, values):
                x_text = value + 1.6 if value < 90 else value - 1.6
                ha = "left" if value < 90 else "right"
                text_color = INK if value < 90 else "white"
                ax.text(
                    x_text,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value}",
                    ha=ha,
                    va="center",
                    fontsize=9.5,
                    color=text_color,
                    fontweight="bold",
                )
        ax.set_title(task, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Success rate (%)")
        ax.set_yticks(y, labels=labels)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.65)
        style_axis(ax)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=legend_labels, loc="upper center", bbox_to_anchor=(0.62, 1.02), ncol=2, fontsize=9)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.82, bottom=0.16, wspace=0.46)
    save_figure(fig, "table_tc_diffusion_policy")


def plot_t_d() -> None:
    labels = ["Robot-only\n$R_{rec}=80, H_{rec}=0$", "EgoRecovery\n$R_{rec}=50, H_{rec}=300$"]
    values = [60, 85]
    colors = [BASELINE_MID, OURS]

    fig, ax = plt.subplots(figsize=(11.6, 4.1))
    fig.patch.set_facecolor(BG)
    y = np.arange(2)
    bars = ax.barh(y, values, height=0.5, color=colors, edgecolor="none")
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Recovery success rate (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.65)
    for bar, value in zip(bars, values):
        ax.text(value - 2.2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", ha="right", va="center", color="white", fontsize=19, fontweight="bold")
    ax.annotate(
        "+25.0 pp",
        xy=(81.5, 0.84),
        xytext=(85.0, 0.56),
        color=OURS_DARK,
        fontsize=15,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": OURS_DARK, "lw": 1.2},
    )
    style_axis(ax)
    ax.tick_params(labelsize=15)
    ax.xaxis.label.set_fontsize(16)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.91, bottom=0.22)
    save_figure(fig, "table_td_cost_matched")


def plot_t_e() -> None:
    x = np.arange(3)
    values = np.array([80, 80, 75])
    labels = ["Clean", "$\\pm$3 frames", "$\\pm$5 frames"]

    fig, ax = plt.subplots(figsize=(11.4, 5.3))
    fig.patch.set_facecolor(BG)
    ax.plot(x, values, color=BASELINE_DARK, linewidth=3.2, marker="o", markersize=12)
    ax.fill_between(x, values, 70, color=BASELINE_SOFT, alpha=0.35)
    for xi, value in zip(x, values):
        ax.text(xi, value + 1.25, f"{value:.0f}", ha="center", va="bottom", fontsize=19, fontweight="bold", color=INK)
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(68, 88)
    ax.set_yticks([70, 75, 80, 85])
    ax.set_ylabel("Recovery success rate (%)")
    ax.set_xlabel("Injected noise in recovery boundary $t_{rec}$")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)
    style_axis(ax)
    ax.tick_params(labelsize=15)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.20)
    save_figure(fig, "table_te_annotation_robustness")


def plot_t_f() -> None:
    metrics = ["Initial SR", "Placement / insertion", "Retreat & re-approach"]
    robot = [[50, 55, 60], [60, 55, 55]]
    ego = [[75, 80, 85], [85, 85, 85]]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1), sharex=True)
    fig.patch.set_facecolor(BG)
    for ax, task, robot_values, ego_values in zip(axes, ["Cup-brush", "Round-disk"], robot, ego):
        y = np.arange(len(metrics))
        for idx, (base, ours) in enumerate(zip(robot_values, ego_values)):
            ax.plot([base, ours], [idx, idx], color=GRID, linewidth=3, zorder=1)
            ax.scatter(base, idx, s=120, color=BASELINE_MID, edgecolor="white", linewidth=1.0, zorder=3)
            ax.scatter(ours, idx, s=130, color=OURS, edgecolor="white", linewidth=1.0, zorder=3)
            ax.text(base - 2.0, idx, f"{base}", ha="right", va="center", fontsize=15, color=BASELINE_DARK)
            ax.text(ours + 2.0, idx, f"{ours}", ha="left", va="center", fontsize=15, color=OURS_DARK, fontweight="bold")
        ax.set_title(task, fontsize=17, fontweight="bold", pad=10)
        ax.set_xlim(35, 103)
        ax.set_xticks([40, 60, 80, 100])
        ax.set_xlabel("Success rate (%)")
        ax.set_yticks(y, labels=metrics)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.65)
        style_axis(ax)
        ax.tick_params(labelsize=14)
        ax.xaxis.label.set_fontsize(15)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=BASELINE_MID, label="Robot-only recovery", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="", color=OURS, label="EgoRecovery", markersize=8),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.67, 1.02), ncol=2, fontsize=13)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.82, bottom=0.16, wspace=0.72)
    save_figure(fig, "table_tf_multiple_failures")


def plot_t_g() -> None:
    annotated_heatmap(
        "table_tg_ood_generalization",
        [
            "Robot recovery only",
            "+ Human recovery from same scenes",
            "+ Human recovery from OOD scenes",
        ],
        ["Cup-brush\nInitial", "Cup-brush\nRecovery", "Round-disk\nInitial", "Round-disk\nRecovery"],
        np.array([[20, 15, 40, 35], [35, 40, 50, 55], [60, 55, 75, 60]]),
        highlight_row=2,
    )


def main() -> None:

    plot_t_c()

    print(f"Rendered rebuttal figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
