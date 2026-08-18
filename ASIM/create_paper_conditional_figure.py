"""Build the compact conditional-performance figure used as manuscript Figure 6."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ASIM2026_analysis"
OUT = ROOT / "ASIM2026_figures" / "paper_results_conditional_mechanism"

SKIES = ["Clear", "Overcast", "Night"]
BLUE = "#087BB5"
LIGHT_BLUE = "#62B5DF"
GREY = "#B8B8B8"
DARK_GREY = "#666666"
ORANGE = "#D95F02"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.4,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.6,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_data():
    density = pd.read_csv(DATA / "deep_density_summary.csv")
    density = density[density["baseline"] == "Rule-based hybrid"].copy()

    saving = (
        density.pivot(index="weather", columns="occupancy_count", values="mean_absolute_saving_W")
        .reindex(SKIES)
        .reindex(columns=range(1, 16))
    )

    outcomes = (
        density.groupby("weather")[["ga_lower_count", "equal_count", "ga_higher_count"]]
        .sum()
        .reindex(SKIES)
    )

    allocation = pd.read_csv(DATA / "deep_power_allocation_summary.csv")
    rule = allocation[allocation["strategy"] == "Rule-based hybrid"].set_index("weather").reindex(SKIES)
    ga = allocation[allocation["strategy"] == "Optimized hybrid"].set_index("weather").reindex(SKIES)
    delta = pd.DataFrame(
        {
            "Ceiling": ga["mean_ceiling_power_W"] - rule["mean_ceiling_power_W"],
            "Task": ga["mean_task_power_W"] - rule["mean_task_power_W"],
            "Total": ga["mean_total_power_W"] - rule["mean_total_power_W"],
        },
        index=SKIES,
    )
    return saving, outcomes, delta


def build():
    saving, outcomes, delta = load_data()

    # 174 mm × 70 mm matches the existing two-column Figure 6 footprint.
    fig = plt.figure(figsize=(6.8504, 2.7559), facecolor="white")
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.65, 1.45, 1.65],
        left=0.075,
        right=0.995,
        bottom=0.20,
        top=0.81,
        wspace=0.46,
    )

    # (a) Density-conditioned paired savings.
    ax = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list("saving", [ORANGE, "#F7F7F7", BLUE])
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=110)
    arr = saving.to_numpy(dtype=float)
    ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(15), labels=np.arange(1, 16))
    ax.set_yticks(np.arange(3), labels=SKIES)
    ax.set_xlabel("Occupancy count")
    ax.set_title("Mean paired saving vs rule hybrid (W)", pad=4)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            color = "white" if value >= 45 else "#333333"
            label = "0" if abs(value) < 0.5 else f"{value:.0f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=5.1, color=color)
    ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
    ax.text(-0.10, 1.17, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.text(
        1.0,
        -0.34,
        "Positive values indicate lower GA power",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.3,
        color=DARK_GREY,
    )

    # (b) Complete case-level outcomes.
    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(3)
    left = np.zeros(3)
    cols = ["ga_lower_count", "equal_count", "ga_higher_count"]
    labels = ["GA lower", "Equal", "GA higher"]
    colors = [BLUE, GREY, ORANGE]
    for col, label, color in zip(cols, labels, colors):
        values = outcomes[col].to_numpy(dtype=float)
        bars = ax.barh(y, values, left=left, height=0.58, color=color, edgecolor="white", linewidth=0.4, label=label)
        for bar, value in zip(bars, values):
            if value >= 3:
                txt_color = "white" if color in (BLUE, ORANGE) else "#222222"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=txt_color,
                    fontweight="bold" if color != GREY else "normal",
                )
        left += values
    ax.set_yticks(y, labels=SKIES)
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("Matched scenarios")
    ax.set_title("Case-level outcome vs rule hybrid", pad=4)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        columnspacing=0.7,
        handlelength=1.1,
        borderaxespad=0,
    )
    ax.text(-0.20, 1.17, "(b)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")

    # (c) Mechanism: GA minus rule-based power allocation.
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(3)
    width = 0.22
    offsets = [-0.31, 0.0, 0.31]
    component_colors = [DARK_GREY, LIGHT_BLUE, BLUE]
    for idx, (component, color) in enumerate(zip(["Ceiling", "Task", "Total"], component_colors)):
        values = delta[component].to_numpy(dtype=float)
        bars = ax.bar(x + offsets[idx], values, width, color=color, edgecolor="#222222", linewidth=0.35, label=component)
        for bar, value in zip(bars, values):
            offset = 2.2 if value >= 0 else -3.2
            va = "bottom" if value >= 0 else "top"
            short_label = f"{value:+.0f}" if abs(value) < 10 else f"{value:+.1f}"
            if abs(value) < 0.5:
                short_label = "0"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                short_label,
                ha="center",
                va=va,
                fontsize=5.2,
            )
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xticks(x, labels=SKIES)
    ax.set_ylim(-95, 14)
    ax.set_ylabel("GA − rule hybrid (W)")
    ax.set_title("Source of the power difference", pad=4)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        columnspacing=0.7,
        handlelength=1.1,
        borderaxespad=0,
    )
    ax.text(-0.17, 1.17, "(c)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")

    fig.savefig(OUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    build()
