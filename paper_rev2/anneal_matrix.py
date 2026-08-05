"""Low-entropy fine-tune stage: continue each completed primary run for a
short budget with a small entropy coefficient, so the deterministic
(argmax) policy sharpens to match the stochastic one.

Motivation: with ent_coef=0.05 the trained policies are strongly
stochastic -- e.g. N4 final model: 90% success sampling actions vs 10%
deterministic.  A brief fine-tune at ent_coef=0.005 collapses the
categorical heads onto their modes without forgetting the behaviour.

    python anneal_matrix.py                 # all completed primary runs
    python anneal_matrix.py --eval-after    # + evaluation of the anneal tag

Resumable: anneal runs whose model.zip exists are skipped.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

MATRIX = [(2, 20), (3, 20), (4, 20), (5, 20), (5, 25)]
SEEDS = [0, 1, 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--from-tag", type=str, default="primary")
    p.add_argument("--tag", type=str, default="anneal")
    p.add_argument("--logdir", type=str, default="runs_paper")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--eval-after", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    for (n, m) in MATRIX:
        name = f"N{n}_M{m}"
        for seed in SEEDS:
            src = HERE / args.logdir / args.from_tag / name / f"seed{seed}" / "model.zip"
            dst_dir = HERE / args.logdir / args.tag / name / f"seed{seed}"
            if not src.exists():
                print(f"== skip {name} seed{seed} (no source model in "
                      f"{args.from_tag})")
                continue
            if (dst_dir / "model.zip").exists():
                print(f"== skip {name} seed{seed} (anneal model exists)")
                continue
            print(f"== anneal {name} seed{seed} steps={args.steps} "
                  f"ent={args.ent_coef} [elapsed "
                  f"{round((time.time()-t0)/60,1)} min]", flush=True)
            cmd = [sys.executable, str(HERE / "train_paper.py"),
                   "--n-robots", str(n), "--map-size", str(m),
                   "--steps", str(args.steps), "--seed", str(seed),
                   "--n-envs", str(args.n_envs),
                   "--ent-coef", str(args.ent_coef),
                   "--init-from", str(src),
                   "--tag", args.tag, "--logdir", args.logdir]
            rc = subprocess.call(cmd, cwd=str(HERE))
            if rc != 0:
                print(f"!! {name} seed{seed} exited {rc}; continuing")

    print(f"anneal matrix done in {round((time.time()-t0)/3600, 2)} h")
    if args.eval_after:
        subprocess.call([sys.executable, str(HERE / "eval_paper.py"),
                         "--tag", args.tag, "--logdir", args.logdir,
                         "--out", "results_anneal"], cwd=str(HERE))


if __name__ == "__main__":
    main()
