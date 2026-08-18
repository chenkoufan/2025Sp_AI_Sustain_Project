from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "ASIM2026_analysis"
FIG_DIR = ROOT / "ASIM2026_figures"
LONG_PATH = ANALYSIS_DIR / "control_strategy_results_long.csv"

WEATHERS = ["Clear", "Overcast", "Night"]
STRATEGIES = ["Zonal PIR ceiling-only", "Rule-based hybrid", "Optimized hybrid"]
BASELINES = STRATEGIES[:2]
WEATHER_COLORS = {"Clear": "#56B4E9", "Overcast": "#E69F00", "Night": "#009E73"}
STRATEGY_COLORS = {
    "Zonal PIR ceiling-only": "#666666",
    "Rule-based hybrid": "#D55E00",
    "Optimized hybrid": "#0072B2",
}
FIG_WIDTH_MM = 174


def configure_style() -> None:
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
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def export(fig: plt.Figure, stem: str) -> None:
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(png, format="png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.tiff", format="tiff", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight",
                facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.svg", format="svg", bbox_inches="tight",
                facecolor="white")
    with Image.open(png) as im:
        im.convert("RGB").save(png, format="PNG", dpi=(600, 600), optimize=True)


def load_and_validate() -> pd.DataFrame:
    long = pd.read_csv(LONG_PATH)
    required = {
        "weather", "occupancy_count", "layout_index", "seat_list", "active_zones",
        "strategy", "ceiling_power_W", "task_power_W", "total_power_W",
    }
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(long) != 675:
        raise ValueError(f"Expected 675 strategy-case rows, found {len(long)}")
    if long[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")
    if not np.allclose(long.ceiling_power_W + long.task_power_W, long.total_power_W):
        raise ValueError("Component power does not reproduce total power")
    counts = long.groupby(["weather", "strategy"], observed=True).size()
    if not (counts == 75).all():
        raise ValueError("Each weather-strategy group must contain 75 cases")
    return long


def build_tables(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    case_keys = ["weather", "occupancy_count", "layout_index", "seat_list", "active_zones"]
    wide = long.pivot_table(index=case_keys, columns="strategy", values="total_power_W").reset_index()
    for baseline in BASELINES:
        wide[f"saving_vs_{baseline}_W"] = wide[baseline] - wide["Optimized hybrid"]

    paired_records = []
    density_records = []
    band_records = []
    for baseline in BASELINES:
        diff_col = f"saving_vs_{baseline}_W"
        for _, row in wide.iterrows():
            diff = float(row[diff_col])
            paired_records.append({
                **{key: row[key] for key in case_keys},
                "baseline": baseline,
                "baseline_power_W": row[baseline],
                "optimized_power_W": row["Optimized hybrid"],
                "absolute_saving_W": diff,
                "outcome": "GA lower" if diff > 1e-8 else "Equal" if abs(diff) <= 1e-8 else "GA higher",
            })

        for (weather, occupancy), group in wide.groupby(["weather", "occupancy_count"]):
            d = group[diff_col]
            base_mean = group[baseline].mean()
            density_records.append({
                "weather": weather,
                "occupancy_count": occupancy,
                "baseline": baseline,
                "n_layouts": len(group),
                "baseline_mean_W": base_mean,
                "optimized_mean_W": group["Optimized hybrid"].mean(),
                "mean_absolute_saving_W": d.mean(),
                "min_absolute_saving_W": d.min(),
                "max_absolute_saving_W": d.max(),
                "relative_saving_pct_from_means": d.mean() / base_mean * 100 if base_mean else np.nan,
                "ga_lower_count": int((d > 1e-8).sum()),
                "equal_count": int((d.abs() <= 1e-8).sum()),
                "ga_higher_count": int((d < -1e-8).sum()),
            })

        band = pd.cut(wide.occupancy_count, [0, 5, 10, 15],
                      labels=["Low (1–5)", "Medium (6–10)", "High (11–15)"])
        temp = wide.assign(density_band=band)
        for (weather, density_band), group in temp.groupby(["weather", "density_band"], observed=True):
            d = group[diff_col]
            base_mean = group[baseline].mean()
            band_records.append({
                "weather": weather,
                "density_band": str(density_band),
                "baseline": baseline,
                "n_cases": len(group),
                "baseline_mean_W": base_mean,
                "optimized_mean_W": group["Optimized hybrid"].mean(),
                "mean_absolute_saving_W": d.mean(),
                "relative_saving_pct_from_means": d.mean() / base_mean * 100 if base_mean else np.nan,
            })

    paired = pd.DataFrame(paired_records)
    density = pd.DataFrame(density_records)
    bands = pd.DataFrame(band_records)

    outcome = (
        paired.groupby(["weather", "baseline", "outcome"], observed=True)
        .size().unstack(fill_value=0).reset_index()
    )
    for col in ["GA lower", "Equal", "GA higher"]:
        if col not in outcome:
            outcome[col] = 0
    overall_pair = (
        paired.groupby(["weather", "baseline"], observed=True)
        .agg(
            n_cases=("absolute_saving_W", "size"),
            mean_saving_W=("absolute_saving_W", "mean"),
            median_saving_W=("absolute_saving_W", "median"),
            q1_saving_W=("absolute_saving_W", lambda x: x.quantile(0.25)),
            q3_saving_W=("absolute_saving_W", lambda x: x.quantile(0.75)),
            min_saving_W=("absolute_saving_W", "min"),
            max_saving_W=("absolute_saving_W", "max"),
        ).reset_index()
    )

    allocation = (
        long.groupby(["weather", "strategy"], observed=True)
        .agg(
            mean_ceiling_power_W=("ceiling_power_W", "mean"),
            mean_task_power_W=("task_power_W", "mean"),
            mean_total_power_W=("total_power_W", "mean"),
        ).reset_index()
    )

    layout_records = []
    for (weather, strategy), group in long.groupby(["weather", "strategy"], observed=True):
        density_groups = group.groupby("occupancy_count", observed=True)
        ranges = density_groups.total_power_W.max() - density_groups.total_power_W.min()
        grand = group.total_power_W.mean()
        density_mean = density_groups.total_power_W.transform("mean")
        total_ss = ((group.total_power_W - grand) ** 2).sum()
        within_ss = ((group.total_power_W - density_mean) ** 2).sum()
        layout_records.append({
            "weather": weather,
            "strategy": strategy,
            "mean_within_density_range_W": ranges.mean(),
            "max_within_density_range_W": ranges.max(),
            "occupancy_at_max_range": int(ranges.idxmax()),
            "within_density_variance_share_pct": within_ss / total_ss * 100 if total_ss else np.nan,
        })
    layout = pd.DataFrame(layout_records)

    return {
        "wide": wide,
        "paired": paired,
        "density": density,
        "bands": bands,
        "outcome": outcome,
        "overall_pair": overall_pair,
        "allocation": allocation,
        "layout": layout,
    }


def plot_paired_evidence(tables: dict[str, pd.DataFrame]) -> None:
    configure_style()
    density = tables["density"]
    outcome = tables["outcome"]
    overall_pair = tables["overall_pair"]
    cmap = LinearSegmentedColormap.from_list("saving", ["#D55E00", "#F7F7F7", "#0072B2"])

    fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, 104 / 25.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    maxima = {BASELINES[0]: 260, BASELINES[1]: 120}
    titles = ["(a) GA saving relative to zonal PIR", "(b) GA saving relative to rule hybrid"]

    for ax, baseline, title in zip(axes, BASELINES, titles):
        pivot = (
            density[density.baseline == baseline]
            .pivot(index="weather", columns="occupancy_count", values="mean_absolute_saving_W")
            .reindex(index=WEATHERS, columns=range(1, 16))
        )
        vmax = maxima[baseline]
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap,
                          norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Occupancy count")
        ax.set_xticks(range(15), range(1, 16))
        ax.set_yticks(range(3), WEATHERS)
        ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(3):
            for col in range(15):
                value = pivot.iloc[row, col]
                color = "white" if abs(value) > vmax * 0.48 else "#222222"
                rounded = int(np.rint(value))
                label = "0" if rounded == 0 else str(rounded)
                ax.text(col, row, label, ha="center", va="center",
                        fontsize=5.3, color=color)
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Mean paired saving (W)")

    ax = fig.add_subplot(grid[1, :])
    rows = []
    labels = []
    means = []
    for weather in WEATHERS:
        for baseline, short in [(BASELINES[0], "vs PIR"), (BASELINES[1], "vs rule")]:
            row = outcome[(outcome.weather == weather) & (outcome.baseline == baseline)].iloc[0]
            rows.append([int(row["GA lower"]), int(row["Equal"]), int(row["GA higher"])])
            labels.append(f"{weather} {short}")
            means.append(float(overall_pair[(overall_pair.weather == weather) &
                                            (overall_pair.baseline == baseline)].mean_saving_W.iloc[0]))
    values = np.asarray(rows)
    y = np.arange(len(labels))
    colors = ["#0072B2", "#BDBDBD", "#D55E00"]
    names = ["GA lower", "Equal", "GA higher"]
    left = np.zeros(len(labels))
    for col, color, name in zip(range(3), colors, names):
        bars = ax.barh(y, values[:, col], left=left, color=color, edgecolor="white",
                       linewidth=0.6, label=name)
        for bar, count in zip(bars, values[:, col]):
            if count >= 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2, str(count),
                        ha="center", va="center", color="white" if col != 1 else "#222222",
                        fontsize=6.3, fontweight="bold")
        left += values[:, col]
    for yi, mean in enumerate(means):
        ax.text(76.5, yi, f"mean {mean:+.1f} W", va="center", fontsize=6.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 92)
    ax.set_xlabel("Matched scenarios (n = 75 per comparison)")
    ax.set_title("(c) Scenario-level dominance and mean paired difference",
                 loc="left", fontweight="bold")
    ax.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(colors, names)],
              ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.16), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.5)
    export(fig, "deep_paired_strategy_evidence")
    plt.close(fig)


def plot_mechanism_and_layout(tables: dict[str, pd.DataFrame]) -> None:
    configure_style()
    allocation = tables["allocation"]
    layout = tables["layout"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_MM / 25.4, 78 / 25.4),
        gridspec_kw={"width_ratios": [1.05, 1]}, layout="constrained"
    )

    x = np.arange(3)
    width = 0.23
    components = [
        ("mean_ceiling_power_W", "Ceiling", "#7A7A7A"),
        ("mean_task_power_W", "Task", "#56B4E9"),
        ("mean_total_power_W", "Total", "#0072B2"),
    ]
    for offset, (column, label, color) in zip([-width, 0, width], components):
        differences = []
        for weather in WEATHERS:
            opt = allocation[(allocation.weather == weather) &
                             (allocation.strategy == "Optimized hybrid")][column].iloc[0]
            rule = allocation[(allocation.weather == weather) &
                              (allocation.strategy == "Rule-based hybrid")][column].iloc[0]
            differences.append(opt - rule)
        bars = ax1.bar(x + offset, differences, width, label=label, color=color,
                       edgecolor="black", linewidth=0.45)
        for bar, value in zip(bars, differences):
            if abs(value) >= 1:
                va = "bottom" if value >= 0 else "top"
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         value + (2 if value >= 0 else -2), f"{value:+.1f}",
                         ha="center", va=va, fontsize=5.8)
    ax1.axhline(0, color="black", linewidth=0.75)
    ax1.set_xticks(x, WEATHERS)
    ax1.set_ylabel("GA minus rule-based hybrid (W)")
    ax1.set_title("(a) Source of the GA–rule difference", loc="left", fontweight="bold")
    ax1.legend(frameon=False, ncol=3, loc="lower left")
    ax1.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    group_x = np.arange(3)
    offsets = [-0.24, 0, 0.24]
    for offset, strategy in zip(offsets, STRATEGIES):
        subset = layout[layout.strategy == strategy].set_index("weather").loc[WEATHERS]
        bars = ax2.bar(group_x + offset, subset.mean_within_density_range_W, 0.22,
                       color=STRATEGY_COLORS[strategy], edgecolor="black", linewidth=0.45)
        for bar, share in zip(bars, subset.within_density_variance_share_pct):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{share:.0f}%", ha="center", va="bottom", fontsize=5.7)
    tick_positions = np.concatenate([group_x + offset for offset in offsets])
    tick_labels = (["PIR"] * 3 + ["Rule"] * 3 + ["GA"] * 3)
    order = np.argsort(tick_positions)
    ax2.set_xticks(tick_positions[order], np.asarray(tick_labels)[order])
    for group_position, weather in zip(group_x, WEATHERS):
        ax2.text(group_position, -0.12, weather, ha="center", va="top",
                 transform=ax2.get_xaxis_transform(), fontweight="bold")
    ax2.set_ylabel("Mean within-density min–max range (W)")
    ax2.set_title("(b) Sensitivity to seating layout", loc="left", fontweight="bold")
    ax2.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.5, -0.20, "Numbers above bars: within-density share of total variation",
             transform=ax2.transAxes, ha="center", va="top", fontsize=5.8,
             color="#444444")
    export(fig, "deep_mechanism_and_layout")
    plt.close(fig)


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    pair = tables["overall_pair"]
    bands = tables["bands"]
    allocation = tables["allocation"]
    layout = tables["layout"]

    def pair_row(weather: str, baseline: str) -> pd.Series:
        return pair[(pair.weather == weather) & (pair.baseline == baseline)].iloc[0]

    def alloc(weather: str, strategy: str) -> pd.Series:
        return allocation[(allocation.weather == weather) &
                          (allocation.strategy == strategy)].iloc[0]

    lines = [
        "# Deep analysis of matched lighting-control results",
        "",
        "## Design boundary",
        "",
        "All comparisons are paired by sky condition, occupancy count, and seating layout. "
        "There are 75 deterministic simulation cases per sky condition (15 occupancy counts × "
        "five layouts). The five layouts are design samples, not independent observations from a "
        "population; therefore the analysis reports descriptive effect sizes and full case counts, "
        "not inferential p values or confidence intervals. Power is reported in W, not energy.",
        "",
        "## Main findings",
        "",
        "1. No control strategy is uniformly best across daylight conditions and comparator choice.",
        "2. GA-derived control strongly reduces power relative to ceiling-only zonal PIR in almost "
        "all cases, but its advantage over the rule-based hybrid is conditional on reduced daylight.",
        "3. Under Night, the GA trades a small increase in task-light power for a much larger decrease "
        "in ceiling-light power.",
        "4. Seating layout explains a material fraction of power variation under Clear and Overcast, "
        "so occupancy count alone is an incomplete control input.",
        "",
        "## Paired performance",
        "",
    ]
    for weather in WEATHERS:
        pir = pair_row(weather, BASELINES[0])
        rule = pair_row(weather, BASELINES[1])
        lines.append(
            f"- **{weather}:** mean paired GA saving was {pir.mean_saving_W:.2f} W versus zonal PIR "
            f"(median {pir.median_saving_W:.2f} W) and {rule.mean_saving_W:.2f} W versus the rule-based "
            f"hybrid (median {rule.median_saving_W:.2f} W)."
        )

    lines.extend(["", "## Mechanism", ""])
    for weather in WEATHERS:
        opt = alloc(weather, "Optimized hybrid")
        rule = alloc(weather, "Rule-based hybrid")
        lines.append(
            f"- **{weather}:** GA minus rule hybrid = "
            f"{opt.mean_ceiling_power_W - rule.mean_ceiling_power_W:+.2f} W ceiling, "
            f"{opt.mean_task_power_W - rule.mean_task_power_W:+.2f} W task, and "
            f"{opt.mean_total_power_W - rule.mean_total_power_W:+.2f} W total."
        )

    lines.extend(["", "## Density bands", ""])
    for baseline in BASELINES:
        lines.append(f"### Relative to {baseline}")
        for weather in WEATHERS:
            subset = bands[(bands.baseline == baseline) & (bands.weather == weather)]
            values = "; ".join(
                f"{r.density_band}: {r.mean_absolute_saving_W:+.1f} W "
                f"({r.relative_saving_pct_from_means:+.1f}%)" for _, r in subset.iterrows()
            )
            lines.append(f"- {weather}: {values}")

    lines.extend(["", "## Spatial sensitivity", ""])
    for weather in WEATHERS:
        parts = []
        for strategy in STRATEGIES:
            row = layout[(layout.weather == weather) & (layout.strategy == strategy)].iloc[0]
            parts.append(
                f"{strategy} {row.mean_within_density_range_W:.1f} W / "
                f"{row.within_density_variance_share_pct:.1f}%"
            )
        lines.append(f"- **{weather}:** " + "; ".join(parts) +
                     " (mean within-density range / share of total variation).")

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The workbooks do not contain the final illuminance and uniformity outputs for each retained "
        "solution. Consequently, these tables establish comparative power and allocation behavior but "
        "do not independently verify that every baseline and optimized case satisfies identical visual-"
        "comfort constraints. The isolated cases in which GA exceeds zonal PIR, and the Clear near-"
        "parity with the rule hybrid, should be checked against the original comfort outputs and repeated "
        "optimizer runs before global-optimality claims are made.",
    ])
    (ANALYSIS_DIR / "deep_results_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    long = load_and_validate()
    tables = build_tables(long)
    tables["paired"].to_csv(ANALYSIS_DIR / "deep_paired_case_differences.csv", index=False)
    tables["density"].to_csv(ANALYSIS_DIR / "deep_density_summary.csv", index=False)
    tables["bands"].to_csv(ANALYSIS_DIR / "deep_density_band_summary.csv", index=False)
    tables["overall_pair"].to_csv(ANALYSIS_DIR / "deep_overall_paired_summary.csv", index=False)
    tables["allocation"].to_csv(ANALYSIS_DIR / "deep_power_allocation_summary.csv", index=False)
    tables["layout"].to_csv(ANALYSIS_DIR / "deep_layout_sensitivity_summary.csv", index=False)
    plot_paired_evidence(tables)
    plot_mechanism_and_layout(tables)
    write_report(tables)
    manifest = {
        "input_strategy_rows": len(long),
        "matched_cases": int(len(tables["wide"])),
        "cases_per_weather": 75,
        "excluded_rows": 0,
        "power_unit": "W",
        "uncertainty_display": "full min–max across five designed layouts where applicable",
        "inference": "descriptive; no p values or population confidence intervals",
        "comfort_verification": "not possible from the supplied workbook columns",
    }
    (ANALYSIS_DIR / "deep_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
