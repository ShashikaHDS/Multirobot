"""Compose the Isaac Sim top-down stills into a lettered sequence figure.

    python compose_sequence.py [--steps 0,9,18,final] [--out figures/isaac_sequence]

Crops each 1080x1080 top-down still to the arena, lays the frames out in
one row with (a)...(d) letters, and writes PDF + PNG for the paper.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", default="media_isaac")
    ap.add_argument("--steps", default="0,9,18,final")
    ap.add_argument("--crop", default="70,70,1010,1010",
                    help="left,top,right,bottom pixel box of the arena")
    ap.add_argument("--out", default="figures/isaac_sequence")
    args = ap.parse_args()

    steps = args.steps.split(",")
    box = tuple(int(v) for v in args.crop.split(","))
    frames = []
    for s in steps:
        name = ("still_top_final.png" if s == "final"
                else f"still_top_step{int(s):03d}.png")
        frames.append((s, Image.open(BASE / args.media / name).crop(box)))

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(7.16, 7.16 / n + 0.25))
    for ax, (s, im) in zip(axes, frames):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        label = "final" if s == "final" else f"step {int(s)}"
        ax.set_xlabel(f"({chr(97 + axes.tolist().index(ax))}) {label}",
                      fontsize=8, labelpad=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.11,
                        wspace=0.03)
    out = BASE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", dpi=300)
    fig.savefig(str(out) + ".png", dpi=200)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
