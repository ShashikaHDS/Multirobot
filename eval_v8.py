"""
Evaluate a trained v8 policy across the paper's test cases.

Test cases reproduced (paper Section IV):
  1. Scalability vs robot count    (vary num_active 2..6)
  2. Scalability vs map size       (vary map_rows / map_cols)
  3. Different obstacle shapes     (different cluster_size_range)
  4. Increasing obstacle density   (vary num_clusters)
  5. Different initial positions   (different seeds for the same map config)

Usage:
    python eval_v8.py --model models_v8/<run>/final_stage3.zip
    python eval_v8.py --model models_v8/<run>/final_stage3.zip --render --n 3
"""

import os
import argparse

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from env_v8_generalized import GeneralizedRendezvousEnv, EnvConfig


def run_episode(model, cfg, seed, render=False):
    env = GeneralizedRendezvousEnv(cfg)
    env.seed(seed)
    obs = env.reset()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if render:
            env.render()
    if render:
        env.close()
    return info


def aggregate(model, cfg, n=10, seed_base=1000, render=False):
    rows = []
    for s in range(n):
        info = run_episode(model, cfg, seed=seed_base + s, render=render)
        d = info["distances"][:info["num_active"]]
        rows.append({
            "success":      bool(info.get("success", False)),
            "steps":        int(info.get("steps", 0)),
            "mean_dist":    float(d.mean()) if len(d) else 0.0,
            "max_dist":     float(d.max()) if len(d) else 0.0,
            "obs_colls":    int(info.get("collide_obstacle", 0)),
            "rob_colls":    int(info.get("collide_robot", 0)),
            "num_active":   int(info["num_active"]),
        })
    df = pd.DataFrame(rows)
    return {
        "success_rate":  float(df["success"].mean()),
        "mean_steps":    float(df["steps"].mean()),
        "mean_distance": float(df["mean_dist"].mean()),
        "max_distance":  float(df["max_dist"].mean()),
        "mean_active":   float(df["num_active"].mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--cells-per-robot", type=int, default=2)
    p.add_argument("--render", action="store_true")
    p.add_argument("--out", default="eval_v8_results.csv")
    args = p.parse_args()

    model = PPO.load(args.model, device="auto")
    K = args.cells_per_robot
    rows = []

    # ---- Test case 1: vary robot count
    for nr in [2, 3, 4, 5, 6]:
        cfg = EnvConfig(map_rows=20, map_cols=20,
                        min_robots=nr, max_robots=nr,
                        cells_per_robot=K)
        m = aggregate(model, cfg, n=args.n, render=args.render)
        rows.append({"case": "scalability_robots", "param": nr, **m})
        print(f"[robots={nr}] {m}")

    # ---- Test case 2: vary map size
    for sz in [20, 25, 30, 40]:
        cfg = EnvConfig(map_rows=sz, map_cols=sz,
                        min_robots=4, max_robots=4,
                        cells_per_robot=K)
        m = aggregate(model, cfg, n=args.n, render=args.render)
        rows.append({"case": "scalability_map", "param": sz, **m})
        print(f"[map={sz}] {m}")

    # ---- Test case 3: different obstacle shapes (cluster sizes)
    for cs_range in [(2, 5), (2, 10), (5, 15), (8, 20)]:
        cfg = EnvConfig(map_rows=20, map_cols=20,
                        min_robots=4, max_robots=4,
                        cells_per_robot=K,
                        cluster_size_range=cs_range)
        m = aggregate(model, cfg, n=args.n, render=args.render)
        rows.append({"case": "obstacle_shape", "param": str(cs_range), **m})
        print(f"[shape={cs_range}] {m}")

    # ---- Test case 4: obstacle density
    for ncl in [3, 5, 8, 12]:
        cfg = EnvConfig(map_rows=20, map_cols=20,
                        num_clusters=ncl,
                        min_robots=4, max_robots=4,
                        cells_per_robot=K)
        m = aggregate(model, cfg, n=args.n, render=args.render)
        rows.append({"case": "obstacle_density", "param": ncl, **m})
        print(f"[density={ncl}] {m}")

    # ---- Test case 5: different initial positions (same config, different seeds)
    cfg = EnvConfig(map_rows=20, map_cols=20,
                    min_robots=4, max_robots=4,
                    cells_per_robot=K)
    for seed_block in range(3):
        m = aggregate(model, cfg, n=args.n,
                      seed_base=2000 + seed_block * 1000,
                      render=args.render)
        rows.append({"case": "initial_positions", "param": seed_block, **m})
        print(f"[positions block {seed_block}] {m}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
