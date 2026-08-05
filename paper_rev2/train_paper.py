"""Train one PPO configuration of the canonical rendezvous env.

Usage (single run):
    python train_paper.py --n-robots 4 --map-size 20 --seed 0 --steps 1000000

Produces under <logdir>/<tag>/N{n}_M{m}/seed{s}/:
    config.json          full provenance (hyperparams, versions, GPU, wall clock)
    model.zip            final model (written only when training completes --
                         its absence marks a crashed/partial run)
    best_model.zip       best checkpoint by mean deterministic-eval reward on
                         a FIXED 10-map evaluation set (same maps every eval)
    checkpoints/         periodic snapshots
    tb/                  TensorBoard logs
    eval/                EvalCallback npz logs

Hyperparameters default to the values stated in the manuscript:
lr 9e-5, ent_coef 0.05, n_steps 2048, batch 64, gamma 0.99, GAE 0.95,
clip 0.2, vf 0.5, max_grad_norm 0.5, MultiInputPolicy with 64-64 tanh heads.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import gymnasium as gym

from env_paper import RendezvousEnv, EnvConfig, RewardConfig

# fixed seeds for the EvalCallback map set: every evaluation scores the
# checkpoint on the SAME 10 maps, so best_model selection is comparable
# across evaluations instead of a noisy argmax over fresh random maps
EVAL_CB_SEEDS = [800_000 + k for k in range(10)]


class EntCoefSchedule(BaseCallback):
    """Linearly interpolate model.ent_coef from start to end over training.

    PPO reads self.ent_coef at every update, so mutating it between
    rollouts implements an entropy schedule (SB3 has no native one).
    High entropy early buys exploration; ~0 late sharpens the policy so
    its argmax matches its behaviour.
    """

    def __init__(self, start: float, end: float, total_steps: int):
        super().__init__()
        self.start, self.end, self.total = start, end, total_steps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        frac = min(1.0, self.num_timesteps / self.total)
        self.model.ent_coef = self.start + frac * (self.end - self.start)


class FixedSeedCycler(gym.Wrapper):
    """Cycles a fixed seed list across resets (ignores incoming seeds)."""

    def __init__(self, env, seeds):
        super().__init__(env)
        self._seeds = list(seeds)
        self._i = 0

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        s = self._seeds[self._i % len(self._seeds)]
        self._i += 1
        return self.env.reset(seed=s, **kwargs)


def make_env(cfg: EnvConfig, seed: int):
    def _thunk():
        env = RendezvousEnv(EnvConfig(**{**cfg.__dict__,
                                         "rewards": RewardConfig(**cfg.rewards.__dict__)}),
                            seed=seed)
        return Monitor(env)
    return _thunk


def make_eval_env(cfg: EnvConfig):
    def _thunk():
        env = RendezvousEnv(EnvConfig(**{**cfg.__dict__,
                                         "rewards": RewardConfig(**cfg.rewards.__dict__)}))
        return Monitor(FixedSeedCycler(env, EVAL_CB_SEEDS))
    return _thunk


def git_commit(cwd: Path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-robots", type=int, default=4)
    p.add_argument("--map-size", type=int, default=20)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=9e-5)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--threshold", type=int, default=16)
    p.add_argument("--tag", type=str, default="primary")
    p.add_argument("--logdir", type=str, default="runs_paper")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-freq", type=int, default=25_000,
                   help="total env steps between deterministic evals")
    p.add_argument("--init-from", type=str, default=None,
                   help="path to a model.zip to continue training from "
                        "(used for the low-entropy fine-tune stage)")
    p.add_argument("--ent-final", type=float, default=None,
                   help="if set, linearly anneal ent_coef from --ent-coef "
                        "to this value over the whole run")
    p.add_argument("--net-width", type=int, default=64,
                   help="hidden width of the two policy/value MLP layers")
    p.add_argument("--step-cost", type=float, default=0.0,
                   help="per-step reward added every step (e.g. -0.1)")
    p.add_argument("--potential-coef", type=float, default=0.0,
                   help="if nonzero, use potential-based area shaping "
                        "coef*(prev_area-new_area) instead of the "
                        "new-best/increase scheme")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    run_dir = here / args.logdir / args.tag / \
        f"N{args.n_robots}_M{args.map_size}" / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    set_random_seed(args.seed)

    cfg = EnvConfig(num_robots=args.n_robots, rows=args.map_size,
                    cols=args.map_size, threshold_area=args.threshold,
                    max_steps=args.max_steps,
                    rewards=RewardConfig(step_cost=args.step_cost,
                                         potential_coef=args.potential_coef))

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env(cfg, seed=args.seed * 1000 + i)
                   for i in range(args.n_envs)])
    # fixed-map eval env for checkpoint selection (see FixedSeedCycler)
    eval_env = DummyVecEnv([make_eval_env(cfg)])

    w = args.net_width
    policy_kwargs = dict(net_arch=dict(pi=[w, w], vf=[w, w]),
                         activation_fn=torch.nn.Tanh)

    if args.init_from:
        model = PPO.load(args.init_from, env=env,
                         learning_rate=args.lr,
                         ent_coef=args.ent_coef,
                         tensorboard_log=str(run_dir / "tb"),
                         seed=args.seed,
                         device=args.device,
                         verbose=1)
    else:
        model = PPO("MultiInputPolicy", env,
                    learning_rate=args.lr,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=args.ent_coef,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=str(run_dir / "tb"),
                    seed=args.seed,
                    device=args.device,
                    verbose=1)

    # SB3's PPO(seed=...) has just silently re-seeded every worker to
    # args.seed + idx, which would make the 3 matrix seeds share most of
    # their per-env map streams (seed0/env1 == seed1/env0 == ...).
    # Re-seed the VecEnv with well-separated streams; these are applied at
    # the first reset inside learn()'s _setup_learn.  Fine-tune runs get a
    # +500 offset so they see fresh maps rather than replaying the
    # primary run's curriculum.
    env.seed(args.seed * 1000 + (500 if args.init_from else 0))

    callbacks = []
    if args.ent_final is not None:
        callbacks.append(EntCoefSchedule(args.ent_coef, args.ent_final,
                                         args.steps))
    callbacks += [
        EvalCallback(eval_env,
                     best_model_save_path=str(run_dir),
                     log_path=str(run_dir / "eval"),
                     eval_freq=max(args.eval_freq // args.n_envs, 1),
                     n_eval_episodes=args.eval_episodes,
                     deterministic=True, render=False),
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1),
                           save_path=str(run_dir / "checkpoints"),
                           name_prefix="ppo"),
    ]

    config = {
        "script": "train_paper.py",
        "argv": vars(args),
        "env_config": {**cfg.__dict__, "rewards": cfg.rewards.__dict__},
        "ppo": {"lr": args.lr, "n_steps": args.n_steps,
                "batch_size": args.batch_size, "n_epochs": 10,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                "ent_coef": args.ent_coef, "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "net_arch": f"pi[{w},{w}] vf[{w},{w}] tanh"},
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None},
        "hostname": platform.node(),
        "git_commit": git_commit(here),
        "init_from": args.init_from,
        "started_unix": time.time(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks,
                progress_bar=False)
    config["wall_clock_sec"] = round(time.time() - t0, 2)
    config["finished_unix"] = time.time()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    model.save(str(run_dir / "model.zip"))
    env.close()
    eval_env.close()
    print(f"DONE {run_dir}  wall={config['wall_clock_sec']}s")


if __name__ == "__main__":
    main()
