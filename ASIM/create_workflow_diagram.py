from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ASIM2026_figures"
OUT.mkdir(exist_ok=True)
STEM = OUT / "balanced_light_control_workflow"


mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def rounded(ax, x, y, w, h, face, edge="#253238", lw=0.9, radius=0.12, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#37474F", style="-", lw=1.15,
          connection="arc3", mutation=10, z=1):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, linestyle=style, color=color,
        connectionstyle=connection, shrinkA=2, shrinkB=2, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=7.2, weight="normal", color="#172126",
          ha="center", va="center", z=4, linespacing=1.15):
    return ax.text(
        x, y, text, ha=ha, va=va, fontsize=size, fontweight=weight,
        color=color, zorder=z, linespacing=linespacing,
    )


def stage_header(ax, number, x, y, title, color, title_size=8.3):
    circle = Circle((x, y), 0.18, facecolor=color, edgecolor="#253238",
                    linewidth=0.8, zorder=5)
    ax.add_patch(circle)
    label(ax, x, y, str(number), size=7.0, weight="bold", color="white", z=6)
    label(ax, x + 0.30, y, title, size=title_size, weight="bold", ha="left")


def main():
    fig, ax = plt.subplots(figsize=(174 / 25.4, 72 / 25.4))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 7.1)
    ax.axis("off")

    # Palette: dark, colorblind-compatible accents with light tints.
    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    PURPLE = "#7A4EAB"
    GREY = "#59666C"
    LIGHT_BLUE = "#E7F2F8"
    LIGHT_GREEN = "#E5F3EE"
    LIGHT_GREY = "#F0F2F3"
    LIGHT_ORANGE = "#FBEDE5"
    LIGHT_YELLOW = "#FFF5D9"
    LIGHT_PURPLE = "#F0EAF7"

    # Stage headers.
    stage_header(ax, 1, 0.40, 6.72, "Scenario inputs", BLUE)
    stage_header(ax, 2, 3.55, 6.72, "Office model", GREEN)
    stage_header(ax, 3, 6.66, 6.72, "Control strategies", ORANGE, 7.8)
    stage_header(ax, 4, 10.18, 6.72, "Evaluation", GREY)
    stage_header(ax, 5, 13.72, 6.72, "Matched comparison", PURPLE, 7.8)

    # Stage 1: inputs.
    rounded(ax, 0.25, 1.25, 2.70, 4.95, LIGHT_BLUE, edge=BLUE, lw=1.1)
    input_items = [
        (5.42, "Sky condition", "Clear / Overcast / Night"),
        (3.88, "Occupancy count", "1-15 occupants"),
        (2.34, "Seat distribution", "5 layouts per count"),
    ]
    for cy, heading, detail in input_items:
        rounded(ax, 0.52, cy - 0.53, 2.16, 1.04, "white", edge=BLUE, lw=0.75)
        label(ax, 1.60, cy + 0.16, heading, size=7.3, weight="bold")
        label(ax, 1.60, cy - 0.19, detail, size=6.7, color="#34434A")
    label(ax, 1.60, 1.55, "75 occupancy-layout patterns\n225 matched cases",
          size=5.9, color=BLUE, weight="bold")

    # Stage 2: model.
    rounded(ax, 3.33, 1.25, 2.70, 4.95, LIGHT_GREEN, edge=GREEN, lw=1.1)
    rounded(ax, 3.60, 4.68, 2.16, 1.12, "white", edge=GREEN, lw=0.75)
    label(ax, 4.68, 5.40, "ClimateStudio\noffice model", size=6.8, weight="bold")
    label(ax, 4.68, 4.88, "Single open-plan room", size=6.2, color="#34434A")
    rounded(ax, 3.60, 3.05, 2.16, 1.12, "white", edge=GREEN, lw=0.75)
    label(ax, 4.68, 3.74, "Background lighting", size=7.2, weight="bold")
    label(ax, 4.68, 3.35, "3 ceiling zones", size=6.7, color="#34434A")
    rounded(ax, 3.60, 1.53, 2.16, 1.04, "white", edge=GREEN, lw=0.75)
    label(ax, 4.68, 2.18, "Task lighting", size=7.2, weight="bold")
    label(ax, 4.68, 1.83, "24 workstation lights", size=6.7, color="#34434A")

    # Stage 3: strategy branches.
    rounded(ax, 6.38, 1.25, 3.02, 4.95, LIGHT_ORANGE, edge=ORANGE, lw=1.1)
    strategy_boxes = [
        (4.78, LIGHT_GREY, GREY, "Zonal PIR ceiling-only", "Active zone: fixed ceiling\nTask lights off"),
        (3.13, "#FFF1E9", ORANGE, "Rule-based hybrid", "Zone ceiling maximum\nOccupied-seat task level"),
        (1.49, "#E8F1F8", BLUE, "GA-derived hybrid", "27 discrete decision variables\nPenalty-based minimization"),
    ]
    for y, face, edge, heading, detail in strategy_boxes:
        rounded(ax, 6.65, y, 2.48, 1.15, face, edge=edge, lw=0.9)
        label(ax, 7.89, y + 0.76, heading, size=6.55, weight="bold")
        label(ax, 7.89, y + 0.34, detail, size=5.75, color="#34434A")

    # Stage 4: simulation, constraints, and power.
    rounded(ax, 9.88, 1.25, 3.13, 4.95, LIGHT_YELLOW, edge=GREY, lw=1.1)
    eval_boxes = [
        (4.75, "Illuminance simulation", "Task and surrounding grids"),
        (3.09, "Visual-comfort check", "Task: 500 lx; U0 >= 0.70\nSurrounding: 300 lx; U0 >= 0.50"),
        (1.48, "Lighting power", "P = 13.2 sum Lc + 1.5 sum Lt"),
    ]
    for y, heading, detail in eval_boxes:
        rounded(ax, 10.15, y, 2.59, 1.14, "white", edge=GREY, lw=0.8)
        label(ax, 11.445, y + 0.77, heading, size=6.7, weight="bold")
        label(ax, 11.445, y + 0.34, detail, size=5.55, color="#34434A")

    # Stage 5: outputs.
    rounded(ax, 13.42, 1.25, 2.88, 4.95, LIGHT_PURPLE, edge=PURPLE, lw=1.1)
    output_items = [
        (5.18, "Power demand (W)"),
        (4.18, "Ceiling / task\nallocation"),
        (3.18, "Savings versus\nbaselines"),
        (2.18, "Occupancy and layout\nsensitivity"),
    ]
    for cy, text in output_items:
        rounded(ax, 13.70, cy - 0.37, 2.32, 0.74, "white", edge=PURPLE, lw=0.75)
        label(ax, 14.86, cy, text, size=6.05, weight="bold")
    label(ax, 14.86, 1.52, "Condition-dependent\ncontrol performance", size=6.7,
          color=PURPLE, weight="bold")

    # Main left-to-right flow.
    arrow(ax, 2.95, 3.72, 3.33, 3.72)
    arrow(ax, 6.03, 3.72, 6.38, 3.72)
    arrow(ax, 9.40, 3.72, 9.88, 3.72)
    arrow(ax, 13.01, 3.72, 13.42, 3.72)

    # Internal evaluation arrows.
    arrow(ax, 11.445, 4.74, 11.445, 4.25, lw=0.85, mutation=8)
    arrow(ax, 11.445, 3.08, 11.445, 2.63, lw=0.85, mutation=8)

    # GA-only iterative feedback path.
    arrow(ax, 10.12, 2.02, 8.98, 1.94, color=ORANGE, style="--", lw=1.0,
          connection="arc3,rad=0.45", mutation=9, z=3)
    label(ax, 9.63, 0.93, "fitness penalty / evolve", size=5.8, color=ORANGE,
          weight="bold")

    # Small legend note.
    label(ax, 8.30, 0.38,
          "All strategies use the same scenarios, comfort criteria, and power coefficients.",
          size=6.5, color="#46555C", weight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.02)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        path = Path(f"{STEM}.{suffix}")
        fig.savefig(path, format=suffix, facecolor="white", transparent=False, **kwargs)
        if suffix == "png":
            with Image.open(path) as image:
                image.convert("RGB").save(path, format="PNG", dpi=(600, 600), optimize=True)
    plt.close(fig)

    manifest = {
        "title": "Balanced Light Control Framework workflow",
        "purpose": "ASIM 2026 conference manuscript methodology diagram",
        "source": "Local manuscript methods and user-specified control rules",
        "layout": "Left-to-right five-stage flow with GA feedback loop",
        "final_width_mm": 174,
        "final_height_mm": 72,
        "formats": ["PNG 600 dpi", "PDF vector", "SVG editable text"],
        "external_services": "none",
        "alt_text": (
            "A five-stage workflow moves from sky condition, occupancy count, and seat distribution "
            "through an office model and three lighting-control strategies to illuminance, comfort, "
            "and power evaluation, followed by matched comparisons. A dashed feedback arrow links "
            "evaluation to the GA-derived strategy to show penalty-based evolution."
        ),
    }
    (OUT / "balanced_light_control_workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
