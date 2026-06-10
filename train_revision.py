"""
Training driver for the revision matrix.

Single-process CLI. Trains one PPO configuration with paper-stated
hyperparameters and writes the model, TensorBoard logs, and a config.json
sidecar so the exact run conditions are preserved.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env_revision import make_env


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-robots", type=int, required=True)
    p.add_argument("--map-size", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--tag", type=str, default="primary")
    p.add_argument("--logdir", type=str, default="logs_revision")
    p.add_argument("--lr", type=float, default=9e-5)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--vec", choices=["subproc", "dummy"], default="subproc")
    args = p.parse_args()

    set_seed(args.seed)

    config_name = f"N{args.n_robots}_M{args.map_size}"
    run_dir = Path(args.logdir) / args.tag / config_name / f"seed{args.seed}"
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env_factories = [
        make_env(args.n_robots, args.map_size, seed=args.seed * 1000 + i)
        for i in range(args.n_envs)
    ]
    VecCls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    env = VecCls(env_factories)

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(tb_dir),
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="ppo",
    )

    t0 = time.time()
    # tb_log_name="PPO" keeps the TB subdir stable across re-runs of the same
    # (tag, config, seed); SB3 still appends _1/_2/... if the dir already
    # exists. We pass an explicit value so the subdir name is predictable for
    # downstream eval/plotting scripts.
    model.learn(
        total_timesteps=args.steps,
        callback=ckpt_cb,
        progress_bar=True,
        tb_log_name="PPO",
    )
    wall = time.time() - t0

    model_path = run_dir / "model.zip"
    model.save(model_path)

    config = {
        "git_commit": git_commit_hash(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_robots": args.n_robots,
        "map_size": args.map_size,
        "seed": args.seed,
        "total_timesteps": args.steps,
        "n_envs": args.n_envs,
        "vec": args.vec,
        "learning_rate": args.lr,
        "ent_coef": args.ent_coef,
        "reward": {
            "area_decrease": 20, "area_increase": -0.5,
            "collision": -5, "goal": 100,
        },
        "wall_clock_sec": round(wall, 2),
        "tag": args.tag,
        "model_path": str(model_path),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    env.close()
    print(f"[done] {config_name} seed={args.seed}  "
          f"wall={wall:.1f}s  saved to {run_dir}")


if __name__ == "__main__":
    main()
