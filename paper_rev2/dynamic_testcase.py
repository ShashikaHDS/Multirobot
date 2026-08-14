"""Test case 06: zero-shot robustness to dynamic obstacles.

Four held-out maps (eval seeds 10000-10003, 20x20, N=4) with the number of
dynamic obstacles rising one per map (map 1 -> k=1 ... map 4 -> k=4).  The
final2m N4 models are evaluated ZERO-SHOT (trained fully static) under the
standard seeded stochastic protocol: 3 training seeds x 5 samples = 15
rollouts per map, censored at the 300-step cap.  Static maps and robot
starts are bit-identical to the static evaluation (obstacles spawn after
placement), so the paired comparison against results_final2m is exact.

Outputs
  results_dynamic/testcase06.csv    per-rollout rows
  results_dynamic/testcase06_summary.csv  per-map aggregate + static pair
  results_dynamic/trajectories.json  one recorded successful rollout per map
  figures/dynamic_paths.{pdf,png}    Fig-9-style two-row panel figure

Run:  python dynamic_testcase.py            (rollouts + figure)
      python dynamic_testcase.py --from-json  (replot only)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAP_SEEDS = [10000, 10001, 10002, 10003]
KS = [1, 2, 3, 4]
N, M = 4, 20
TRAIN_SEEDS = [0, 1, 2]
SAMPLES = 5

ROBOT_COLORS = ["#e41a1c", "#7cca62", "#2a6fb0", "#f28e2b"]
DYN_COLOR = "#8c1aff"
OBST_COLOR = "#111111"
GRID_COLOR = "#cccccc"


def record_rollout(model, env, map_seed, sample_k, record=False):
    """One seeded stochastic rollout; mirrors eval_paper.rollout_ppo."""
    import torch
    torch.manual_seed((map_seed * 1000 + sample_k) % (2 ** 31))
    obs, _ = env.reset(seed=map_seed)
    traj = None
    if record:
        traj = {
            "static_map": env._static_map.tolist(),
            "starts": env.positions.tolist(),
            "dyn_starts": env._dyn_pos.tolist(),
            "robot_paths": [[tuple(p)] for p in env.positions.tolist()],
            "dyn_paths": [[tuple(p)] for p in env._dyn_pos.tolist()],
        }
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=False)
        obs, r, terminated, truncated, info = env.step(action)
        if record:
            for i, p in enumerate(env.positions.tolist()):
                traj["robot_paths"][i].append(tuple(p))
            for i, p in enumerate(env._dyn_pos.tolist()):
                traj["dyn_paths"][i].append(tuple(p))
    ok = bool(info.get("is_success", False))
    return ok, env.step_count, int(env.distances.sum()), \
        int(info.get("obstacle_collisions", 0)), traj


def run(out_dir: Path):
    from stable_baselines3 import PPO
    from env_paper import RendezvousEnv, EnvConfig

    out_dir.mkdir(exist_ok=True)
    rows = []
    trajs = {}
    models = {s: PPO.load(str(HERE / "runs_paper" / "final2m" / f"N{N}_M{M}"
                              / f"seed{s}" / "model.zip"), device="cpu")
              for s in TRAIN_SEEDS}
    for mi, (map_seed, k) in enumerate(zip(MAP_SEEDS, KS), start=1):
        cfg = EnvConfig(num_robots=N, rows=M, cols=M,
                        num_dynamic_obstacles=k)
        env = RendezvousEnv(cfg)
        first_success = None
        for s in TRAIN_SEEDS:
            for j in range(SAMPLES):
                ok, steps, dist, _, _ = record_rollout(
                    models[s], env, map_seed, j)
                rows.append({"map": mi, "map_seed": map_seed, "num_dyn": k,
                             "train_seed": s, "sample": j, "success": int(ok),
                             "steps": steps, "total_distance": dist})
                if ok and first_success is None:
                    first_success = (s, j)
        # re-run the first successful rollout with recording for the figure
        if first_success is None:
            first_success = (TRAIN_SEEDS[0], 0)   # fall back: record a failure
        s, j = first_success
        _, _, _, _, traj = record_rollout(models[s], env, map_seed, j,
                                          record=True)
        traj["map"] = mi
        traj["num_dyn"] = k
        traj["recorded_rollout"] = {"train_seed": s, "sample": j}
        trajs[str(mi)] = traj
        succ = np.mean([r["success"] for r in rows if r["map"] == mi])
        print(f"map {mi} (seed {map_seed}, k={k}): success {succ:.2f} "
              f"over {len(TRAIN_SEEDS) * SAMPLES} rollouts")

    with open(out_dir / "testcase06.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # paired static success on the same maps from the frozen headline results
    static = {}
    fin = HERE / "results_final2m" / "results.csv"
    if fin.exists():
        with open(fin) as f:
            for r in csv.DictReader(f):
                if (r["method"] == "ppo" and int(r["n"]) == N
                        and int(r["m"]) == M
                        and int(r["episode"]) in range(len(MAP_SEEDS))):
                    static.setdefault(int(r["episode"]) + 1, []).append(
                        float(r["success"]))
    with open(out_dir / "testcase06_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "map_seed", "num_dyn", "success_dyn",
                    "success_static", "steps_mean", "dist_mean"])
        for mi, (map_seed, k) in enumerate(zip(MAP_SEEDS, KS), start=1):
            sel = [r for r in rows if r["map"] == mi]
            w.writerow([mi, map_seed, k,
                        round(float(np.mean([r["success"] for r in sel])), 4),
                        round(float(np.mean(static.get(mi, [np.nan]))), 4),
                        round(float(np.mean([r["steps"] for r in sel])), 1),
                        round(float(np.mean([r["total_distance"] for r in sel])), 1)])
    with open(out_dir / "trajectories.json", "w") as f:
        json.dump(trajs, f)
    return trajs


def plot(trajs, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.4))
    for col in range(4):
        t = trajs[str(col + 1)]
        static = np.array(t["static_map"])
        k = t["num_dyn"]
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(np.where(static == 1, 1.0, 0.0), cmap="gray_r",
                      vmin=0, vmax=1, origin="upper")
            ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, M, 1), minor=True)
            ax.grid(which="minor", color=GRID_COLOR, linewidth=0.4)
            ax.tick_params(which="both", bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_color("#555555")
        # top row: initial positions
        ax = axes[0, col]
        ax.set_title(f"Map 0{col + 1}  ($k={k}$)", fontsize=11)
        for i, p in enumerate(t["starts"]):
            ax.add_patch(plt.Rectangle((p[1] - 0.5, p[0] - 0.5), 1, 1,
                                       color=ROBOT_COLORS[i]))
        for p in t["dyn_starts"]:
            ax.add_patch(plt.Rectangle((p[1] - 0.5, p[0] - 0.5), 1, 1,
                                       color=DYN_COLOR))
        # bottom row: travelled paths
        ax = axes[1, col]
        for i, path in enumerate(t["dyn_paths"]):
            arr = np.array(path)
            for cell in {tuple(c) for c in path}:
                ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5),
                                           1, 1, color=DYN_COLOR, alpha=0.18,
                                           linewidth=0))
            ax.plot(arr[:, 1], arr[:, 0], color=DYN_COLOR, linewidth=1.2,
                    linestyle=(0, (2, 2)), alpha=0.9)
            ax.add_patch(plt.Rectangle((arr[-1, 1] - 0.5, arr[-1, 0] - 0.5),
                                       1, 1, color=DYN_COLOR))
        for i, path in enumerate(t["robot_paths"]):
            arr = np.array(path)
            ax.plot(arr[:, 1], arr[:, 0], color=ROBOT_COLORS[i],
                    linewidth=1.6)
            ax.add_patch(plt.Rectangle((arr[-1, 1] - 0.5, arr[-1, 0] - 0.5),
                                       1, 1, color=ROBOT_COLORS[i]))
        ax.set_xlabel(f"({chr(96 + col + 1)})", fontsize=11)

    handles = [Patch(color=ROBOT_COLORS[i], label=f"Robot {i + 1}")
               for i in range(4)]
    handles.append(Patch(color=DYN_COLOR, label="Dynamic obstacle"))
    handles.append(Line2D([0], [0], color=DYN_COLOR, linestyle=(0, (2, 2)),
                          label="Obstacle trail"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    print("saved", fig_path.with_suffix(".pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_dynamic")
    ap.add_argument("--from-json", action="store_true")
    args = ap.parse_args()
    out = HERE / args.out
    if args.from_json:
        trajs = json.load(open(out / "trajectories.json"))
    else:
        trajs = run(out)
    plot(trajs, HERE / "figures" / "dynamic_paths")
