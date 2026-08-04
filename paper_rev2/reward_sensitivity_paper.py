"""One-at-a-time reward sensitivity analysis on the canonical env.

Exactly the protocol stated in the manuscript (Section IV, Test case 06):
4 reward parameters x 5 perturbation levels, each trained for --steps
(default 50k) with all other parameters at their Table III defaults,
evaluated on a fixed held-out set with the *unperturbed* reward metrics.

    python reward_sensitivity_paper.py --steps 50000 --seeds 0 1 2

Writes reward_sensitivity.csv incrementally (safe to interrupt/resume:
existing (param, level, seed) rows are skipped).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from env_paper import RendezvousEnv, EnvConfig, RewardConfig

SWEEP = {
    "area_decrease":   [5.0, 10.0, 20.0, 40.0, 80.0],
    "area_increase":   [-2.0, -1.0, -0.5, -0.1, 0.0],
    "collide_obstacle": [-20.0, -10.0, -5.0, -1.0, 0.0],
    "collide_robot":    [-20.0, -10.0, -5.0, -1.0, 0.0],
}

N_ROBOTS = 4
MAP = 20
EVAL_EPISODES = 20
EVAL_SEED_BASE = 42_000


def train_one(param: str, level: float, seed: int, steps: int, n_envs: int):
    rewards = RewardConfig()
    setattr(rewards, param, level)
    cfg = EnvConfig(num_robots=N_ROBOTS, rows=MAP, cols=MAP, rewards=rewards)

    set_random_seed(seed)
    env = DummyVecEnv([
        (lambda i=i: Monitor(RendezvousEnv(
            EnvConfig(num_robots=N_ROBOTS, rows=MAP, cols=MAP,
                      rewards=RewardConfig(**rewards.__dict__)),
            seed=seed * 1000 + i)))
        for i in range(n_envs)])

    model = PPO("MultiInputPolicy", env,
                learning_rate=9e-5, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                   activation_fn=torch.nn.Tanh),
                seed=seed, verbose=0)
    model.learn(total_timesteps=steps)
    env.close()
    return model


def evaluate(model):
    """Deterministic rollouts on held-out maps, default reward env."""
    env = RendezvousEnv(EnvConfig(num_robots=N_ROBOTS, rows=MAP, cols=MAP))
    succ, steps, dist, obs_c, rob_c = [], [], [], [], []
    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        terminated = truncated = False
        oc = rc = 0
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            oc += info["obstacle_collisions"]
            rc += info["robot_collisions"]
        succ.append(bool(info.get("is_success", False)))
        steps.append(env.step_count)
        dist.append(info.get("total_distance", 0))
        obs_c.append(oc)
        rob_c.append(rc)
    env.close()
    return {"success_rate": float(np.mean(succ)),
            "steps_mean": float(np.mean(steps)),
            "total_dist_mean": float(np.mean(dist)),
            "obstacle_collisions_mean": float(np.mean(obs_c)),
            "robot_collisions_mean": float(np.mean(rob_c))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--out", type=str, default="results/reward_sensitivity.csv")
    args = p.parse_args()

    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = ["param", "level", "seed", "steps", "success_rate", "steps_mean",
              "total_dist_mean", "obstacle_collisions_mean",
              "robot_collisions_mean"]
    done = set()
    if out.exists():
        with open(out) as f:
            for row in csv.DictReader(f):
                done.add((row["param"], float(row["level"]), int(row["seed"])))
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    total = sum(len(v) for v in SWEEP.values()) * len(args.seeds)
    i = 0
    for param, levels in SWEEP.items():
        for level in levels:
            for seed in args.seeds:
                i += 1
                if (param, level, seed) in done:
                    print(f"[{i}/{total}] skip {param}={level} seed{seed}")
                    continue
                print(f"[{i}/{total}] train {param}={level} seed{seed} "
                      f"({args.steps} steps)")
                model = train_one(param, level, seed, args.steps, args.n_envs)
                met = evaluate(model)
                with open(out, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(
                        {"param": param, "level": level, "seed": seed,
                         "steps": args.steps, **met})
                print(f"          success={met['success_rate']:.2f}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
