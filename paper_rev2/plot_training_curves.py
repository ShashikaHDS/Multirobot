"""Training-curve figure (paper Fig. 4): mean episode reward and mean
episode length vs environment steps, for every configuration of a tag.

Reads the SB3 TensorBoard event files under
runs_paper/<tag>/N*_M*/seed*/tb/**/events.*  (push them from the training
machine with `git add -f paper_rev2/runs_paper/<tag>/*/seed*/tb/`).

    python plot_training_curves.py [--tag final2] [--out figures/training_curves]

Design: hue encodes fleet size (consistent with the sensitivity figure);
line style encodes map size (solid 20x20, dashed 25x25); per-config line =
mean over the 3 seeds, shaded band = min-max across seeds; light rolling
smoothing (window 5 logged points) noted here so the caption can state it.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator)

# hue = fleet size (N3/N4/N5 identical to plot_sensitivity.py); N2 added
HUES = {2: "#8250c4", 3: "#2a78d6", 4: "#eb6834", 5: "#1baf7a"}
STYLE = {20: "-", 25: (0, (5, 2))}          # map size -> line style
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

TAGS = [("rollout/ep_rew_mean", "Mean episode reward"),
        ("rollout/ep_len_mean", "Mean episode length (steps)")]


def load_scalar(run_dir: Path, tag: str):
    """Concatenate a scalar tag across all event files of one run."""
    pts = []
    for ev in sorted(run_dir.glob("tb/**/events.*")):
        acc = EventAccumulator(str(ev.parent), size_guidance={"scalars": 0})
        acc.Reload()
        if tag in acc.Tags().get("scalars", []):
            pts += [(s.step, s.value) for s in acc.Scalars(tag)]
    pts.sort()
    if not pts:
        return None, None
    xs, ys = zip(*pts)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def smooth(y, w=5):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, kernel, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="final2m")
    p.add_argument("--logdir", type=str, default="runs_paper")
    p.add_argument("--out", type=str, default="figures/training_curves")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--column", action="store_true",
                   help="single-column figure, episode-length panel only")
    p.add_argument("--stacked", action="store_true",
                   help="single-column figure, both panels stacked")
    p.add_argument("--narrow", action="store_true",
                   help="single-column figure, both panels side by side")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    base = here / args.logdir / args.tag

    # gather (n, m) -> seed -> (xs, ys) per metric
    data = {t: defaultdict(dict) for t, _ in TAGS}
    configs = []
    for cfg_dir in sorted(base.glob("N*_M*")):
        n = int(cfg_dir.name.split("_")[0][1:])
        m = int(cfg_dir.name.split("_")[1][1:])
        configs.append((n, m))
        for seed_dir in sorted(cfg_dir.glob("seed*")):
            for t, _ in TAGS:
                xs, ys = load_scalar(seed_dir, t)
                if xs is not None:
                    data[t][(n, m)][seed_dir.name] = (xs, ys)
    configs = sorted(set(configs))
    if not configs or not any(data[t] for t, _ in TAGS):
        raise SystemExit(
            f"no TensorBoard scalars found under {base} — push the tb/ "
            f"folders from the training machine first")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "text.color": TEXT, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    if args.column:
        TAGS[:] = TAGS[1:]                 # episode-length panel only
        fig, ax_one = plt.subplots(1, 1, figsize=(3.45, 2.5))
        axes = [ax_one]
    elif args.stacked:
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.1))
    elif args.narrow:
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.1))

    for ax, (tag, ylabel) in zip(axes, TAGS):
        for (n, m) in configs:
            runs = data[tag].get((n, m), {})
            if not runs:
                continue
            # common step grid across seeds (interpolate)
            grid = min((xs[-1] for xs, _ in runs.values()))
            xg = np.linspace(0, grid, 200)
            ymat = np.vstack([np.interp(xg, xs, smooth(ys, args.smooth))
                              for xs, ys in runs.values()])
            mean = ymat.mean(axis=0)
            color = HUES[n]
            ax.fill_between(xg, ymat.min(axis=0), ymat.max(axis=0),
                            color=color, alpha=0.13, linewidth=0)
            ax.plot(xg, mean, color=color, linestyle=STYLE[m],
                    linewidth=1.0 if args.narrow else 1.7,
                    label=f"N = {n}" + (f" ({m}×{m})" if m != 20 else ""))
        ax.set_xlabel("Environment steps")
        if args.narrow:
            # horizontal panel title instead of a rotated ylabel, so the
            # axes reclaim the label's horizontal space
            ax.set_title(ylabel.replace("Mean episode", "Episode")
                         .replace(" (steps)", ""),
                         fontsize=7, loc="left", pad=3)
        else:
            ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e6:g}M" if v else "0"))
        if args.stacked:
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        if args.narrow:
            ax.xaxis.set_major_locator(plt.FixedLocator(
                [0, 0.5e6, 1e6, 1.5e6, 2e6]))
            ax.xaxis.set_minor_locator(plt.FixedLocator(
                [0.25e6, 0.75e6, 1.25e6, 1.75e6]))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(labelsize=6, pad=1.5)
            ax.tick_params(which="minor", length=2)
            ax.xaxis.label.set_size(7)
            ax.xaxis.labelpad = 1.5

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3 if (args.column or args.stacked or args.narrow) else 5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.22 if args.narrow else (0.14 if args.column else (0.10 if args.stacked else 0.06)), 1, 1))

    out = here / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
