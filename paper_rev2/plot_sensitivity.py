"""Render the reward-sensitivity figure (paper Test case 06) from
results/reward_sensitivity.csv.

Two metric rows x 4 reward parameters:
  row 1  success rate            (saturates at N=3, informative at N=5)
  row 2  total distance travelled (energy proxy; discriminates everywhere,
         including where success is at ceiling -- this is what answers the
         "no sensitivity at small N" objection)

    python plot_sensitivity.py [--csv path] [--out figures/sensitivity]
                               [--metric2 total_dist_mean|steps_mean]

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

METRIC_LABEL = {
    "success_rate": "Success rate",
    "total_dist_mean": "Fleet distance (cells)",
    "steps_mean": "Steps to converge",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/reward_sensitivity.csv")
    p.add_argument("--out", type=str, default="figures/sensitivity")
    p.add_argument("--metric2", type=str, default="total_dist_mean",
                   choices=["total_dist_mean", "steps_mean"])
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    metrics = ["success_rate", args.metric2]
    data = defaultdict(dict)          # (metric, param, n) -> {level: value}
    with open(here / args.csv) as f:
        for row in csv.DictReader(f):
            for m in metrics:
                data[(m, row["param"], int(row["n_robots"]))][
                    float(row["level"])] = float(row[m])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7.5, "text.color": TEXT, "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))

    for col, (param, label, default) in enumerate(PANELS):
        levels = sorted({x for m in metrics for n in SERIES
                         for x in data.get((m, param, n), {})})
        # discrete sweep levels -> equal spacing (avoids label collisions
        # from the non-uniform numeric spacing of the tested values)
        pos = {lv: i for i, lv in enumerate(levels)}
        for rowi, metric in enumerate(metrics):
            ax = axes[rowi, col]
            for n, (color, marker) in SERIES.items():
                pts = sorted(data.get((metric, param, n), {}).items())
                if not pts:
                    continue
                ax.plot([pos[x] for x, _ in pts], [y for _, y in pts],
                        color=color, marker=marker, markersize=3.8,
                        linewidth=1.5, label=f"N = {n}", clip_on=False,
                        markeredgecolor="white", markeredgewidth=0.5)
            ax.axvline(pos[default], color=MUTED, linestyle=(0, (4, 3)),
                       linewidth=0.9, zorder=0)
            if rowi == 0:
                ax.text(pos[default], 1.05, "default", ha="center",
                        va="bottom", fontsize=6.5, color=MUTED,
                        transform=ax.get_xaxis_transform())
                ax.set_ylim(0, 1.0)
            else:
                ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel(METRIC_LABEL[metric])
            if levels:
                ax.set_xlim(-0.35, len(levels) - 0.65)
                ax.set_xticks(list(pos.values()))
                ax.set_xticklabels([f"{v:g}" for v in levels])
            ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = here / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
