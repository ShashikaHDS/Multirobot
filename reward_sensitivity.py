"""
One-at-a-time sensitivity analysis for the reward components.

For each reward parameter in SWEEP_LEVELS, train short PPO models at several
levels (paper default included) and evaluate on a held-out seed set with the
PAPER reward (so the evaluation metric is consistent across runs). Output is a
CSV plus a summary plot of success_rate and mean_episode_length per level.

Tradeoff: 5 params * 5 levels = 25 trainings * --steps each. Default 100k steps
takes ~2-4 hours on an RTX 2060.
"""

import os
import time
import argparse

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env_v8_generalized import GeneralizedRendezvousEnv, EnvConfig, PAPER_REWARDS
from feature_extractor_v8 import MultiRobotCNNExtractor


SWEEP_LEVELS = {
    "area_decrease":   [5.0, 10.0, 20.0, 40.0, 80.0],
    "area_increase":   [-2.0, -1.0, -0.5, -0.1, 0.0],
    "collide_obstacle": [-20.0, -10.0, -5.0, -1.0, 0.0],
    "collide_robot":   [-20.0, -10.0, -5.0, -1.0, 0.0],
    "goal":            [25.0, 50.0, 100.0, 200.0, 500.0],
}

EVAL_EPISODES = 20
EVAL_SEED_BASE = 42_000


def make_train_env(reward_overrides, eval_phase=False):
    def _thunk():
        cfg = EnvConfig(
            map_rows=15, map_cols=15,
            min_robots=2, max_robots=4,
            cells_per_robot=2,
            max_steps=300,
        )
        cfg.rewards = dict(PAPER_REWARDS)
        if not eval_phase:
            cfg.rewards.update(reward_overrides)
        return GeneralizedRendezvousEnv(cfg)
    return _thunk


def evaluate(model, n_episodes=EVAL_EPISODES):
    env = make_train_env({}, eval_phase=True)()
    successes = 0
    ep_lens, mean_dists, obs_colls, rob_colls = [], [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        terminated = truncated = False
        oc = rc = 0
        steps = 0
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            oc += info.get("collide_obstacle", 0)
            rc += info.get("collide_robot", 0)
            steps += 1
        if info.get("success"):
            successes += 1
        ep_lens.append(steps)
        d = info["distances"][:info["num_active"]]
        mean_dists.append(float(d.mean()) if len(d) else 0.0)
        obs_colls.append(oc)
        rob_colls.append(rc)
    env.close()
    return dict(
        success_rate=successes / n_episodes,
        mean_episode_length=float(np.mean(ep_lens)),
        mean_distance=float(np.mean(mean_dists)),
        mean_obs_collisions=float(np.mean(obs_colls)),
        mean_robot_collisions=float(np.mean(rob_colls)),
    )


def train_one(reward_overrides, total_timesteps, seed):
    env = DummyVecEnv([make_train_env(reward_overrides)])
    policy_kwargs = dict(
        features_extractor_class=MultiRobotCNNExtractor,
        features_extractor_kwargs=dict(features_dim=128, robot_embed_dim=64),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    model = PPO(
        "MultiInputPolicy", env, verbose=0,
        ent_coef=0.05, learning_rate=9e-5,
        policy_kwargs=policy_kwargs,
        seed=seed, device="auto",
    )
    model.learn(total_timesteps=total_timesteps)
    metrics = evaluate(model)
    env.close()
    del model
    return metrics


def plot(df, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return
    metrics = ["success_rate", "mean_episode_length",
               "mean_distance", "mean_obs_collisions"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, metrics):
        for param in SWEEP_LEVELS:
            d = df[df.param == param].sort_values("level")
            ax.plot(d["level"], d[metric], "o-", label=param)
        ax.set_title(metric)
        ax.set_xlabel("reward level")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--out", default="reward_sensitivity_results.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only-param", default=None,
                   help="Sweep only a single parameter (debug).")
    args = p.parse_args()

    rows = []
    t0 = time.time()
    items = SWEEP_LEVELS.items()
    if args.only_param:
        items = [(args.only_param, SWEEP_LEVELS[args.only_param])]

    for param, levels in items:
        for level in levels:
            print(f"\n=== {param}={level} ===")
            metrics = train_one({param: level}, total_timesteps=args.steps,
                                seed=args.seed)
            row = {"param": param, "level": level, **metrics,
                   "elapsed_min": (time.time() - t0) / 60}
            print(row)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
    plot(df, args.out.replace(".csv", ".png"))


if __name__ == "__main__":
    main()
