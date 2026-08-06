"""Render the reward-sensitivity figure (paper Test case 06) from
results/reward_sensitivity.csv: 4 panels (one per reward parameter),
success rate vs perturbation level, one line per robot count, the
paper's default value marked.

    python plot_sensitivity.py [--csv path] [--out figures/sensitivity]

Design notes: categorical palette (validated colorblind-safe trio),
distinct markers per series so the figure survives grayscale printing,
recessive grid, direct value ticks at the exact sweep levels, default
level marked with a neutral dashed rule + label.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical trio (all-pairs colorblind-safe, light surface)
SERIES = {3: ("#2a78d6", "o"), 4: ("#eb6834", "s"), 5: ("#1baf7a", "^")}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

PANELS = [
    ("potential_coef", "Shaping coefficient  $c_\\phi$", 0.5),
    ("step_cost", "Step cost  $r_{step}$", -0.1),
    ("collide_obstacle", "Obstacle collision  $r_{obs}$", -5.0),
    ("collide_robot", "Robot collision  $r_{rob}$", -5.0),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/reward_sensitivity.csv")
    p.add_argument("--out", type=str, default="figures/sensitivity")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    data = defaultdict(dict)          # (param, n) -> {level: success}
    with open(here / args.csv) as f:
        for row in csv.DictReader(f):
            data[(row["param"], int(row["n_robots"]))][
                float(row["level"])] = float(row["success_rate"])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7.5, "text.color": TEXT, "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))

    for ax, (param, label, default) in zip(axes.flat, PANELS):
        for n, (color, marker) in SERIES.items():
            pts = sorted(data.get((param, n), {}).items())
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            ax.plot(xs, ys, color=color, marker=marker, markersize=4.5,
                    linewidth=1.6, label=f"N = {n}", clip_on=False,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.axvline(default, color=MUTED, linestyle=(0, (4, 3)),
                   linewidth=0.9, zorder=0)
        ax.text(default, 1.045, "default", ha="center", va="bottom",
                fontsize=6.5, color=MUTED, transform=ax.get_xaxis_transform())
        ax.set_xlabel(label)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Success rate")
        levels = sorted({x for n in SERIES for x in data.get((param, n), {})})
        if levels:
            ax.set_xticks(levels)
            ax.set_xticklabels([f"{v:g}" for v in levels])
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    out = here / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
