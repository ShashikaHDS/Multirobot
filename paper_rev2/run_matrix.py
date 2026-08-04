"""Sequential orchestrator for the paper's 15-run training matrix:
N in {2,3,4,5} on 20x20 and N=5 on 25x25, seeds {0,1,2}.

Resumable: a run whose model.zip already exists is skipped, so the script
can be re-launched after an interruption.

    python run_matrix.py                # full matrix
    python run_matrix.py --only N4_M20  # one configuration
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

MATRIX = [
    # (n_robots, map_size, steps)
    (2, 20, 1_000_000),
    (3, 20, 1_000_000),
    (4, 20, 1_500_000),
    (5, 20, 1_500_000),
    (5, 25, 2_000_000),
]
SEEDS = [0, 1, 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="primary")
    p.add_argument("--logdir", type=str, default="runs_paper")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--only", type=str, default=None,
                   help="restrict to one config, e.g. N4_M20")
    p.add_argument("--eval-after", action="store_true",
                   help="run eval_paper.py when the matrix finishes")
    args = p.parse_args()

    t0 = time.time()
    for (n, m, steps) in MATRIX:
        name = f"N{n}_M{m}"
        if args.only and name != args.only:
            continue
        for seed in SEEDS:
            run_dir = HERE / args.logdir / args.tag / name / f"seed{seed}"
            if (run_dir / "model.zip").exists():
                print(f"== skip {name} seed{seed} (model.zip exists)")
                continue
            print(f"== train {name} seed{seed} steps={steps} "
                  f"[elapsed {round((time.time()-t0)/60,1)} min]", flush=True)
            cmd = [sys.executable, str(HERE / "train_paper.py"),
                   "--n-robots", str(n), "--map-size", str(m),
                   "--steps", str(steps), "--seed", str(seed),
                   "--n-envs", str(args.n_envs),
                   "--tag", args.tag, "--logdir", args.logdir]
            rc = subprocess.call(cmd, cwd=str(HERE))
            if rc != 0:
                print(f"!! {name} seed{seed} exited {rc}; continuing")

    print(f"matrix done in {round((time.time()-t0)/3600, 2)} h")
    if args.eval_after:
        subprocess.call([sys.executable, str(HERE / "eval_paper.py"),
                        "--tag", args.tag, "--logdir", args.logdir],
                        cwd=str(HERE))


if __name__ == "__main__":
    main()
