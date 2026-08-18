"""Create paired strategy evidence with result-centered terminology."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np

from deep_results_analysis import BASELINES, WEATHERS, build_tables, load_and_validate


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ASIM2026_figures" / "optimized_paired_strategy_evidence"
WIDTH_MM = 174


def configure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "legend.fontsize": 6.7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def export(fig):
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def main():
    configure_style()
    long = load_and_validate()
    tables = build_tables(long)
    density = tables["density"]
    outcome = tables["outcome"]
    overall_pair = tables["overall_pair"]

    cmap = LinearSegmentedColormap.from_list("saving", ["#D55E00", "#F7F7F7", "#0072B2"])
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 104 / 25.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    maxima = {BASELINES[0]: 260, BASELINES[1]: 120}
    titles = [
        "(a) Optimized hybrid saving relative to zonal PIR",
        "(b) Optimized hybrid saving relative to rule-based hybrid",
    ]

    for ax, baseline, title in zip(axes, BASELINES, titles):
        pivot = (
            density[density.baseline == baseline]
            .pivot(index="weather", columns="occupancy_count", values="mean_absolute_saving_W")
            .reindex(index=WEATHERS, columns=range(1, 16))
        )
        vmax = maxima[baseline]
        heatmap = ax.imshow(
            pivot.to_numpy(), aspect="auto", cmap=cmap,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Occupancy count")
        ax.set_xticks(range(15), range(1, 16))
        ax.set_yticks(range(3), WEATHERS)
        ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(3):
            for column in range(15):
                value = pivot.iloc[row, column]
                color = "white" if abs(value) > vmax * 0.48 else "#222222"
                rounded = int(np.rint(value))
                ax.text(column, row, "0" if rounded == 0 else str(rounded),
                        ha="center", va="center", fontsize=5.3, color=color)
        colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.045, pad=0.02)
        colorbar.set_label("Mean paired saving (W)")

    ax = fig.add_subplot(grid[1, :])
    rows, labels, means = [], [], []
    for weather in WEATHERS:
        for baseline, short in [(BASELINES[0], "vs PIR"), (BASELINES[1], "vs rule")]:
            row = outcome[(outcome.weather == weather) & (outcome.baseline == baseline)].iloc[0]
            rows.append([int(row["GA lower"]), int(row["Equal"]), int(row["GA higher"])])
            labels.append(f"{weather} {short}")
            means.append(float(overall_pair[
                (overall_pair.weather == weather) & (overall_pair.baseline == baseline)
            ].mean_saving_W.iloc[0]))

    values = np.asarray(rows)
    y = np.arange(len(labels))
    colors = ["#0072B2", "#BDBDBD", "#D55E00"]
    names = ["Optimized lower", "Equal", "Optimized higher"]
    left = np.zeros(len(labels))
    for column, color, name in zip(range(3), colors, names):
        bars = ax.barh(y, values[:, column], left=left, color=color,
                       edgecolor="white", linewidth=0.6, label=name)
        for bar, count in zip(bars, values[:, column]):
            if count >= 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2, str(count),
                        ha="center", va="center",
                        color="white" if column != 1 else "#222222",
                        fontsize=6.3, fontweight="bold")
        left += values[:, column]
    for position, mean in enumerate(means):
        ax.text(76.5, position, f"mean {mean:+.1f} W", va="center", fontsize=6.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 92)
    ax.set_xlabel("Matched scenarios (n = 75 per comparison)")
    ax.set_title("(c) Scenario-level outcomes and mean paired difference",
                 loc="left", fontweight="bold", y=1.16)
    ax.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(colors, names)],
              ncol=3, loc="lower center", bbox_to_anchor=(0.58, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.5)

    export(fig)
    plt.close(fig)
    print(f"Created {OUT.name} from {len(long)} strategy-case rows with no exclusions")


if __name__ == "__main__":
    main()
