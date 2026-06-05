"""
Evaluation pipeline.

Loads each trained PPO policy under logs_revision/<tag>/<config>/<seed>/
and rolls it out for K episodes per evaluation map. Runs the A* baseline
on the same maps with the same seeds. Writes a single results.csv with one
row per (method, config, map, seed).
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from stable_baselines3 import PPO

from env_revision import RendezvousEnv
from astar_baseline import run_astar_episode


def discover_runs(logdir: Path, tag: str) -> List[Dict]:
    runs = []
    tag_dir = logdir / tag
    if not tag_dir.exists():
        return runs
    for cfg_dir in sorted(tag_dir.iterdir()):
        if not cfg_dir.is_dir():
            continue
        for seed_dir in sorted(cfg_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            cfg = seed_dir / "config.json"
            model = seed_dir / "model.zip"
            if cfg.exists() and model.exists():
                with open(cfg) as f:
                    config = json.load(f)
                runs.append({"config": config, "model_path": str(model)})
    return runs


def rollout_ppo(model: PPO, env: RendezvousEnv) -> Dict:
    obs, _ = env.reset()
    total = 0
    per_robot = np.zeros(env.n_robots, dtype=np.int32)
    success = False
    steps = 0
    while steps < env.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        prev = env._positions.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        for i in range(env.n_robots):
            if not np.array_equal(env._positions[i], prev[i]):
                per_robot[i] += 1
                total += 1
        steps += 1
        if terminated:
            success = True
            break
        if truncated:
            break
    return {
        "success": int(success),
        "steps": steps,
        "total_distance": int(total),
        "max_distance": int(per_robot.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="logs_revision")
    p.add_argument("--tag", type=str, default="primary")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", type=str, default="results_revision/results.csv")
    p.add_argument("--baseline", action="store_true", default=True,
                   help="Also run the A* baseline on the same maps.")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(Path(args.logdir), args.tag)
    if not runs:
        print(f"No runs found under {args.logdir}/{args.tag}")
        return
    print(f"Found {len(runs)} trained policies under tag '{args.tag}'.")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "config", "n_robots", "map_size", "seed", "episode",
            "success", "steps", "total_distance", "max_distance",
            "inference_ms_per_step",
        ])

        for run in runs:
            cfg = run["config"]
            n_robots = cfg["n_robots"]
            map_size = cfg["map_size"]
            train_seed = cfg["seed"]
            config_name = f"N{n_robots}_M{map_size}"

            model = PPO.load(run["model_path"], device="cuda" if torch.cuda.is_available() else "cpu")

            for ep in range(args.episodes):
                eval_seed = 10_000 + ep  # held-out eval seeds
                env = RendezvousEnv(n_robots=n_robots, map_size=map_size)

                # PPO
                env.reset(seed=eval_seed)
                t0 = time.time()
                m = rollout_ppo(model, env)
                infer_ms = 1000.0 * (time.time() - t0) / max(m["steps"], 1)
                w.writerow([
                    "ppo", config_name, n_robots, map_size, train_seed, ep,
                    m["success"], m["steps"], m["total_distance"], m["max_distance"],
                    round(infer_ms, 3),
                ])

                # A* baseline on same map seed
                if args.baseline:
                    env_b = RendezvousEnv(n_robots=n_robots, map_size=map_size)
                    env_b.reset(seed=eval_seed)
                    mb = run_astar_episode(env_b)
                    w.writerow([
                        "astar", config_name, n_robots, map_size, train_seed, ep,
                        mb["success"], mb["steps"], mb["total_distance"], mb["max_distance"],
                        "",
                    ])

            print(f"  done: {config_name} seed={train_seed}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
