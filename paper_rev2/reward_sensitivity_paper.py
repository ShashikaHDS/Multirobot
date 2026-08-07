"""One-at-a-time reward sensitivity analysis on the FINAL recipe,
for N in {3, 4, 5} robots (paper Test case 06 + reviewer request).

Base configuration (the paper's final reward/training recipe):
    potential shaping 0.5 * dArea, step cost -0.1, goal +100,
    collisions -5 each; ent_coef annealed 0.05 -> 0; lr 3e-4;
    64-64 tanh MultiInputPolicy; 20x20 map.

Sweep: 4 parameters x 5 levels x 3 robot counts (the base point is
trained once per N and reused across parameters):
    potential_coef   0.1  0.25  [0.5]  1.0  2.0
    step_cost       -0.5  -0.2  [-0.1] -0.05 0.0
    collide_obstacle -20  -10   [-5]   -1    0
    collide_robot    -20  -10   [-5]   -1    0

Each point trains --steps (default 200k) with seed 0 and is evaluated
with the paper protocol: 5 seeded stochastic rollouts on each of the 20
held-out maps (censored means). Metrics: success rate, steps, total /
max distance, Jain fairness. Incremental CSV -> safe to interrupt and
resume (existing (param, level, n) rows are skipped).

    python reward_sensitivity_paper.py            # full sweep (~5 h)
    python plot_sensitivity.py                    # figure from the CSV
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env_paper import RendezvousEnv, EnvConfig, RewardConfig
from train_paper import EntCoefSchedule

BASE = dict(potential_coef=0.5, step_cost=-0.1,
            collide_obstacle=-5.0, collide_robot=-5.0, goal=100.0)

# Levels are chosen relative to the reward scale (goal = +100, typical
# per-step shaping |c_phi * dA| ~ 1-5) so that each parameter is driven
# past the point where it must dominate the return: the sweep is meant to
# expose BOTH the robust plateau and its boundaries, not only the plateau.
#   c_phi   0.02  -> shaping ~ absent (near sparse-reward problem)
#           5.0   -> a single step's shaping rivals the goal reward
#   r_step  -5    -> 20 steps of dawdling cost the whole goal reward
#   r_obs   -100  -> one obstacle contact cancels the goal reward
#   r_rob   -100  -> one robot contact cancels the goal reward
SWEEP = {
    "potential_coef":   [0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    "step_cost":        [-5.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0],
    "collide_obstacle": [-100.0, -50.0, -20.0, -10.0, -5.0, -1.0, 0.0],
    "collide_robot":    [-100.0, -50.0, -20.0, -10.0, -5.0, -1.0, 0.0],
}

MAP = 20
EVAL_MAPS = 20
EVAL_SAMPLES = 5
EVAL_SEED_BASE = 10_000
LR = 3e-4
ENT0, ENT1 = 0.05, 0.0


def make_rewards(param, level):
    vals = dict(BASE)
    vals[param] = level
    return RewardConfig(**vals)


def train_one(n_robots, rewards, steps, seed, n_envs):
    cfg = EnvConfig(num_robots=n_robots, rows=MAP, cols=MAP, rewards=rewards)
    set_random_seed(seed)

    def thunk(i):
        return lambda: Monitor(RendezvousEnv(
            EnvConfig(num_robots=n_robots, rows=MAP, cols=MAP,
                      rewards=RewardConfig(**rewards.__dict__)),
            seed=seed * 1000 + i))

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = vec_cls([thunk(i) for i in range(n_envs)])
    model = PPO("MultiInputPolicy", env,
                learning_rate=LR, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=ENT0, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                   activation_fn=torch.nn.Tanh),
                seed=seed, verbose=0)
    env.seed(seed * 1000)          # undo SB3's silent env re-seed
    model.learn(total_timesteps=steps,
                callback=EntCoefSchedule(ENT0, ENT1, steps))
    env.close()
    return model


def evaluate(model, n_robots):
    """Always evaluated with the DEFAULT reward metrics, so perturbed runs
    are scored on the same yardstick.  Collision counts are reported
    because a weaker collision penalty can raise success while producing a
    policy that is physically unacceptable on real hardware -- the choice
    of penalty is a success/contact trade-off, not a pure maximisation."""
    env = RendezvousEnv(EnvConfig(num_robots=n_robots, rows=MAP, cols=MAP))
    succ, cf, st, td, md, jn, oc, rc = [], [], [], [], [], [], [], []
    for s in range(EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_MAPS):
        for k in range(EVAL_SAMPLES):
            torch.manual_seed((s * 1000 + k) % (2 ** 31))
            obs, _ = env.reset(seed=s)
            term = trunc = False
            info = {}
            n_obs = n_rob = 0
            while not (term or trunc):
                a, _ = model.predict(obs, deterministic=False)
                obs, r, term, trunc, info = env.step(a)
                n_obs += info["obstacle_collisions"]
                n_rob += info["robot_collisions"]
            ok = bool(info.get("is_success"))
            succ.append(ok)
            # collision-free success: the headline metric must not be
            # improvable by weakening the collision penalty, otherwise the
            # sweep rewards policies that bump through obstacles (which
            # real hardware cannot do). Obstacle contacts are the
            # safety-critical ones; robot-robot contacts are reported
            # separately as a coordination metric.
            cf.append(ok and n_obs == 0)
            st.append(env.step_count)
            td.append(info["total_distance"])
            md.append(info["max_distance"])
            oc.append(n_obs)
            rc.append(n_rob)
            d = env.distances.astype(float)
            jn.append((d.sum() ** 2) / (len(d) * (d ** 2).sum())
                      if d.sum() > 0 else 1.0)
    env.close()
    return {"success_rate": float(np.mean(succ)),
            "cf_success_rate": float(np.mean(cf)),
            "steps_mean": float(np.mean(st)),
            "total_dist_mean": float(np.mean(td)),
            "max_dist_mean": float(np.mean(md)),
            "jain_mean": float(np.mean(jn)),
            "obs_collisions_mean": float(np.mean(oc)),
            "robot_collisions_mean": float(np.mean(rc))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--robots", type=int, nargs="+", default=[4, 3, 5])
    p.add_argument("--params", type=str, nargs="+", default=None,
                   choices=list(SWEEP), help="restrict to these parameters")
    p.add_argument("--out", type=str, default="results/reward_sensitivity.csv")
    args = p.parse_args()

    sweep = {k: v for k, v in SWEEP.items()
             if args.params is None or k in args.params}

    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = ["param", "level", "n_robots", "seed", "steps", "success_rate",
              "cf_success_rate", "steps_mean", "total_dist_mean",
              "max_dist_mean", "jain_mean", "obs_collisions_mean",
              "robot_collisions_mean"]
    done = set()
    if out.exists():
        with open(out) as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            old_rows = list(reader)
        for row in old_rows:
            done.add((row["param"], float(row["level"]),
                      int(row["n_robots"])))
        if old_fields != fields:
            # schema migration: rewrite with the current header, leaving
            # newly added metrics blank for rows collected before they existed
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in old_rows:
                    w.writerow({k: row.get(k, "") for k in fields})
            print(f"migrated {out.name} to the current schema "
                  f"({len(old_rows)} existing rows preserved)")
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def write(param, level, n, met):
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(
                {"param": param, "level": level, "n_robots": n,
                 "seed": args.seed, "steps": args.steps, **met})

    total = sum(len(v) for v in sweep.values()) * len(args.robots)
    i = 0
    default_cache = {}          # n -> metrics of the all-default point
    for n in args.robots:
        for param, levels in sweep.items():
            for level in levels:
                i += 1
                if (param, level, n) in done:
                    print(f"[{i}/{total}] skip {param}={level} N{n}",
                          flush=True)
                    continue
                is_default = (level == BASE[param])
                if is_default and n in default_cache:
                    write(param, level, n, default_cache[n])
                    print(f"[{i}/{total}] reuse default N{n} for {param}",
                          flush=True)
                    continue
                print(f"[{i}/{total}] train {param}={level} N{n} "
                      f"({args.steps} steps)", flush=True)
                model = train_one(n, make_rewards(param, level),
                                  args.steps, args.seed, args.n_envs)
                met = evaluate(model, n)
                if is_default:
                    default_cache[n] = met
                write(param, level, n, met)
                print(f"          success={met['success_rate']:.2f} "
                      f"dist={met['total_dist_mean']:.0f}", flush=True)
    print(f"SENSITIVITY_SWEEP_DONE -> {out}")


if __name__ == "__main__":
    main()
