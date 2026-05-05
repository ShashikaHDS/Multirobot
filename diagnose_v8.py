"""
One-shot diagnostic for a v8 training run.

Loads each stage checkpoint and evaluates it on 3 fixed configs (easy / paper /
big) using the SAME config for every stage so the comparison is apples-to-apples.
The output table makes 3 patterns visible:
  - rising then falling success%       -> catastrophic forgetting in curriculum
  - all near 0%                        -> env/reward/training is broken
  - last stage strong, earlier stages weaker -> normal (expected)

Usage on the 5090 PC (with the venv active):
    python diagnose_v8.py --run v8_1777656757
"""

import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from env_v8_generalized import GeneralizedRendezvousEnv, EnvConfig


def evaluate(model, cfg, n=20, seed_base=1000):
    successes = 0
    steps_list = []
    obs_coll_list = []
    for s in range(n):
        env = GeneralizedRendezvousEnv(cfg)
        obs, _ = env.reset(seed=seed_base + s)
        terminated = truncated = False
        info = {}
        oc = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            oc += info.get("collide_obstacle", 0)
        if info.get("success"):
            successes += 1
        steps_list.append(info.get("steps", 0))
        obs_coll_list.append(oc)
    return (successes / n,
            float(np.mean(steps_list)),
            float(np.mean(obs_coll_list)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="run dir name, e.g. v8_1777656757")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--cells-per-robot", type=int, default=2)
    args = p.parse_args()

    run_dir = os.path.join("models_v8", args.run)
    if not os.path.isdir(run_dir):
        print(f"NOT FOUND: {run_dir}")
        return

    K = args.cells_per_robot
    configs = [
        ("easy_10x10_4r",
         EnvConfig(map_rows=10, map_cols=10, max_robots=6,
                   min_robots=4, max_active=4, cells_per_robot=K,
                   max_steps=200, num_clusters=3, cluster_size_range=(2, 5))),
        ("paper_20x20_4r",
         EnvConfig(map_rows=20, map_cols=20, max_robots=6,
                   min_robots=4, max_active=4, cells_per_robot=K,
                   max_steps=400, num_clusters=5, cluster_size_range=(2, 10))),
        ("big_30x30_4r",
         EnvConfig(map_rows=30, map_cols=30, max_robots=6,
                   min_robots=4, max_active=4, cells_per_robot=K,
                   max_steps=600, num_clusters=7, cluster_size_range=(3, 12))),
    ]

    print(f"\n=== Diagnostic for {run_dir}  (n={args.n} per cell) ===")
    header = f"{'ckpt':<8} {'config':<20} {'success%':>9} {'mean_steps':>11} {'obs_coll':>9}"
    print(header)
    print("-" * len(header))

    for stage in range(4):
        ckpt = os.path.join(run_dir, f"final_stage{stage}.zip")
        if not os.path.exists(ckpt):
            print(f"stage{stage}: missing")
            continue
        try:
            model = PPO.load(ckpt, device="auto")
        except Exception as e:
            print(f"stage{stage}: load failed -- {type(e).__name__}: {e}")
            continue
        for name, cfg in configs:
            sr, ms, oc = evaluate(model, cfg, n=args.n)
            print(f"stage{stage:<3} {name:<20} {sr*100:>8.1f}% {ms:>11.0f} {oc:>9.1f}")
        del model

    print()
    print("Interpretation:")
    print("  - if success% rises then falls across stages: curriculum forgetting")
    print("  - if all rows are near 0%: env / reward / training fundamentally broken")
    print("  - if final stage strongest, earlier stages weaker: expected")


if __name__ == "__main__":
    main()
