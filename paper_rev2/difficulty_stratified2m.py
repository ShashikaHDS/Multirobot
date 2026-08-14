"""Difficulty-stratified frontier-vs-RL comparison.

Maps are selected by a DISCLOSED generation property (obstacle load),
never by outcome: four density levels via the generator's cluster-size
knob, 20 pre-registered seeds per level (seeds 20000-20019, fixed before
any evaluation). Both methods run on every map of every level -- nothing
is filtered afterwards.

For each map we also record the number of obstacle-free 4x4 squares in
the TRUE map ("valid goal regions"): the mechanism variable. Hypothesis:
one-shot-commitment planners degrade as valid regions become scarce,
while the learned policy remains robust.

N=4, 20x20, PPO = final2 models (3 seeds x 3 samples per map),
frontier = both danger variants (stronger per map reported).

    python difficulty_stratified.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_paper import RendezvousEnv, EnvConfig, FREE
from frontier_baseline import run_frontier_episode

TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
FRONTIER_C, PPO_C = "#b3453b", "#0b0b0b"

LEVELS = [("sparse", 5, (2, 5)), ("baseline", 5, (2, 10)),
          ("large", 5, (5, 15)), ("dense", 8, (8, 20))]
SEEDS = [20_000 + i for i in range(20)]
N, M = 4, 20
PPO_SAMPLES = 3


def n_valid_squares(grid, side=4):
    R, C = grid.shape
    obs = (grid != FREE).astype(np.int32)
    ii = np.zeros((R + 1, C + 1), dtype=np.int32)
    ii[1:, 1:] = obs.cumsum(0).cumsum(1)
    s = ii[side:, side:] - ii[:-side, side:] - ii[side:, :-side] \
        + ii[:-side, :-side]
    return int((s == 0).sum())


def rollout_ppo(model, env, seed, k):
    import torch
    torch.manual_seed((seed * 1000 + k) % (2 ** 31))
    obs, _ = env.reset(seed=seed)
    term = trunc = False
    info = {}
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=False)
        obs, r, term, trunc, info = env.step(a)
    return {"success": bool(info.get("is_success")),
            "steps": env.step_count,
            "total_distance": int(env.distances.sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results_difficulty")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    models = [PPO.load(str(here / "runs_paper" / "final2m" / f"N{N}_M{M}"
                           / f"seed{ts}" / "model.zip"), device="cpu")
              for ts in (0, 1, 2)]

    rows = []
    for (level, ncl, csr) in LEVELS:
        cfg = EnvConfig(num_robots=N, rows=M, cols=M, num_clusters=ncl,
                        cluster_size_range=csr)
        env = RendezvousEnv(cfg)
        for seed in SEEDS:
            env.reset(seed=seed)
            nvs = n_valid_squares(env.grid_map)
            obs_cells = int((env.grid_map != FREE).sum())
            # PPO
            succ, st, td = [], [], []
            for model in models:
                for k in range(PPO_SAMPLES):
                    met = rollout_ppo(model, env, seed, k)
                    succ.append(met["success"])
                    st.append(met["steps"])
                    td.append(met["total_distance"])
            rows.append({"level": level, "seed": seed, "method": "ppo",
                         "success": float(np.mean(succ)),
                         "steps": float(np.mean(st)),
                         "total_distance": float(np.mean(td)),
                         "n_valid_squares": nvs,
                         "obstacle_cells": obs_cells})
            # frontier: stronger variant per map
            best = None
            for ud in (True, False):
                met = run_frontier_episode(env, seed=seed, use_danger=ud)
                cand = (float(met["success"]), -met["steps"], met)
                if best is None or cand[:2] > best[:2]:
                    best = cand
            met = best[2]
            rows.append({"level": level, "seed": seed, "method": "frontier",
                         "success": float(met["success"]),
                         "steps": float(met["steps"]),
                         "total_distance": float(met["total_distance"]),
                         "n_valid_squares": nvs,
                         "obstacle_cells": obs_cells})
        env.close()
        for meth in ("ppo", "frontier"):
            sel = [r for r in rows if r["level"] == level
                   and r["method"] == meth]
            print(f"{level:>9} {meth:>8}: success "
                  f"{np.mean([r['success'] for r in sel]):.2f}  "
                  f"valid-squares "
                  f"{np.mean([r['n_valid_squares'] for r in sel]):.0f}",
                  flush=True)

    fields = ["level", "seed", "method", "success", "steps",
              "total_distance", "n_valid_squares", "obstacle_cells"]
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ------------------------------ figure ----------------------------- #
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": TEXT, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.7))

    xs = np.arange(len(LEVELS))
    for meth, color, ls in (("ppo", PPO_C, "-"),
                            ("frontier", FRONTIER_C, (0, (4, 2)))):
        ys = [np.mean([r["success"] for r in rows
                       if r["level"] == lv and r["method"] == meth])
              for (lv, _, _) in LEVELS]
        a1.plot(xs, ys, color=color, linestyle=ls, marker="o",
                markersize=4.5, linewidth=1.7,
                label="PPO" if meth == "ppo" else "Frontier",
                markeredgecolor="white", markeredgewidth=0.5)
    a1.set_xticks(xs)
    a1.set_xticklabels([lv for (lv, _, _) in LEVELS])
    a1.set_xlabel("Obstacle load (generator setting)")
    a1.set_ylabel("Success rate")
    a1.set_ylim(0, 1.02)
    a1.legend(frameon=False)

    # mechanism: success vs valid-goal-region count (binned)
    bins = [(0, 20), (20, 40), (40, 70), (70, 200)]
    for meth, color, ls in (("ppo", PPO_C, "-"),
                            ("frontier", FRONTIER_C, (0, (4, 2)))):
        bx, by = [], []
        for (lo, hi) in bins:
            sel = [r["success"] for r in rows if r["method"] == meth
                   and lo <= r["n_valid_squares"] < hi]
            if sel:
                bx.append((lo + hi) / 2)
                by.append(np.mean(sel))
        a2.plot(bx, by, color=color, linestyle=ls, marker="o",
                markersize=4.5, linewidth=1.7,
                markeredgecolor="white", markeredgewidth=0.5)
    a2.set_xlabel("Valid 4×4 goal regions in true map")
    a2.set_ylabel("Success rate")
    a2.set_ylim(0, 1.02)
    for ax in (a1, a2):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(here / "figures" / "difficulty_stratified") + ext,
                    dpi=300, bbox_inches="tight")
    print(f"wrote {out_dir}/results.csv and figures/difficulty_stratified")


if __name__ == "__main__":
    main()
