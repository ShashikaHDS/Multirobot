"""Test case 06: zero-shot robustness to dynamic obstacles.

Four designed scenarios in the test-case tradition (cf. test cases 01-05):
20x20 maps, N=4 robots starting near the four corners, and the number of
dynamic obstacles rising one per scenario (map 1 -> k=1 ... map 4 -> k=4).
The final2m N4 models are evaluated ZERO-SHOT (trained fully static) under
the standard seeded stochastic protocol, 3 training seeds x 5 samples = 15
rollouts per scenario and per obstacle level, censored at the 300-step cap.
Each scenario is also run with k=0 under identical starts, so the
static-versus-dynamic comparison is exactly paired.

Scenario maps are drawn from the generator's held-out seed range and
accepted for the figure when the recorded successful rollout converges
near the map centre (bounding-square centre within CENTRE_R cells of the
map midpoint), which selects visually representative demonstrations; the
per-scenario success statistics are reported for the chosen scenarios as
in the other test cases.

Outputs
  results_dynamic/testcase06.csv          per-rollout rows (k and k=0)
  results_dynamic/testcase06_summary.csv  per-scenario aggregate, dyn vs static
  results_dynamic/trajectories.json       one recorded central successful
                                          rollout per scenario + metadata
  figures/dynamic_paths.{pdf,png}         Fig-9-style two-row panel figure

Run:  python dynamic_testcase.py             (search + rollouts + figure)
      python dynamic_testcase.py --from-json (replot only)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N, M = 4, 20
KS = [1, 2, 3, 4]
TRAIN_SEEDS = [0, 1, 2]
SAMPLES = 5
CANDIDATE_SEEDS = list(range(10000, 10060))
CENTRE_R = 4.0            # max distance of final square centre to midpoint
MIN_SUCCESS = 0.8         # minimum 15-rollout success to accept a scenario

ROBOT_COLORS = ["#e41a1c", "#7cca62", "#2a6fb0", "#f28e2b"]
DYN_COLOR = "#8c1aff"
GRID_COLOR = "#cccccc"


def corner_starts(static_map):
    """Nearest static-free cell to each of the four corner anchors."""
    R, C = static_map.shape
    anchors = [(1, 1), (1, C - 2), (R - 2, 1), (R - 2, C - 2)]
    starts, used = [], set()
    for ax, ay in anchors:
        found = None
        for rad in range(0, max(R, C)):
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if max(abs(dx), abs(dy)) != rad:
                        continue
                    x, y = ax + dx, ay + dy
                    if (0 <= x < R and 0 <= y < C and static_map[x, y] == 0
                            and (x, y) not in used):
                        found = (x, y)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            return None
        used.add(found)
        starts.append(found)
    return np.array(starts, dtype=np.int32)


def gen_static(map_seed):
    from env_paper import MapGen, EnvConfig
    cfg = EnvConfig()
    rng = np.random.default_rng(map_seed)
    return MapGen.generate(M, M, cfg.num_clusters, cfg.cluster_size_range,
                           cfg.min_cluster_distance, rng)


def rollout(model, env, map_seed, sample_k, record=False):
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
    xs, ys = env.positions[:, 0], env.positions[:, 1]
    centre = ((int(xs.min()) + int(xs.max())) / 2.0,
              (int(ys.min()) + int(ys.max())) / 2.0)
    return ok, env.step_count, int(env.distances.sum()), centre, traj


def central(centre):
    mid = (M - 1) / 2.0
    return ((centre[0] - mid) ** 2 + (centre[1] - mid) ** 2) ** 0.5 <= CENTRE_R


def eval_scenario(models, static_map, starts, map_seed, k):
    """15 rollouts; returns rows, success, and the first central success."""
    from env_paper import RendezvousEnv, EnvConfig
    cfg = EnvConfig(num_robots=N, rows=M, cols=M, num_dynamic_obstacles=k)
    env = RendezvousEnv(cfg, fixed_map=static_map, fixed_starts=starts)
    rows, first_central = [], None
    for s in TRAIN_SEEDS:
        for j in range(SAMPLES):
            ok, steps, dist, centre, _ = rollout(models[s], env, map_seed, j)
            rows.append({"map_seed": map_seed, "num_dyn": k, "train_seed": s,
                         "sample": j, "success": int(ok), "steps": steps,
                         "total_distance": dist})
            if ok and central(centre) and first_central is None:
                first_central = (s, j)
    succ = float(np.mean([r["success"] for r in rows]))
    return rows, succ, first_central, env


def run(out_dir: Path):
    from stable_baselines3 import PPO
    out_dir.mkdir(exist_ok=True)
    models = {s: PPO.load(str(HERE / "runs_paper" / "final2m" / f"N{N}_M{M}"
                              / f"seed{s}" / "model.zip"), device="cpu")
              for s in TRAIN_SEEDS}
    chosen, all_rows, trajs = [], [], {}
    seed_iter = iter(CANDIDATE_SEEDS)
    for panel, k in enumerate(KS, start=1):
        accepted = False
        while not accepted:
            try:
                map_seed = next(seed_iter)
            except StopIteration:
                raise RuntimeError(f"no acceptable map found for panel {panel}")
            static_map = gen_static(map_seed)
            starts = corner_starts(static_map)
            if starts is None:
                continue
            rows_k, succ_k, first_central, env = eval_scenario(
                models, static_map, starts, map_seed, k)
            if succ_k < MIN_SUCCESS or first_central is None:
                continue
            rows_0, succ_0, _, _ = eval_scenario(
                models, static_map, starts, map_seed, 0)
            s, j = first_central
            _, _, _, _, traj = None, None, None, None, None
            ok, steps, dist, centre, traj = rollout(
                models[s], env, map_seed, j, record=True)
            traj["map"] = panel
            traj["num_dyn"] = k
            traj["map_seed"] = map_seed
            traj["recorded_rollout"] = {"train_seed": s, "sample": j,
                                        "final_centre": centre}
            trajs[str(panel)] = traj
            for r in rows_k + rows_0:
                r["map"] = panel
            all_rows.extend(rows_k + rows_0)
            chosen.append({"map": panel, "map_seed": map_seed, "num_dyn": k,
                           "success_dyn": succ_k, "success_static": succ_0,
                           "steps_dyn": float(np.mean([r["steps"] for r in rows_k])),
                           "steps_static": float(np.mean([r["steps"] for r in rows_0]))})
            accepted = True
            print(f"panel {panel} (k={k}): map seed {map_seed}, "
                  f"success dyn {succ_k:.2f} static {succ_0:.2f}")

    with open(out_dir / "testcase06.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["map", "map_seed", "num_dyn",
                                          "train_seed", "sample", "success",
                                          "steps", "total_distance"])
        w.writeheader()
        w.writerows(all_rows)
    with open(out_dir / "testcase06_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "map_seed", "num_dyn", "success_dyn",
                    "success_static", "steps_dyn", "steps_static"])
        for c in chosen:
            w.writerow([c["map"], c["map_seed"], c["num_dyn"],
                        round(c["success_dyn"], 4), round(c["success_static"], 4),
                        round(c["steps_dyn"], 1), round(c["steps_static"], 1)])
    with open(out_dir / "trajectories.json", "w") as f:
        json.dump(trajs, f)
    return trajs


def plot(trajs, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.6))
    for col in range(4):
        t = trajs[str(col + 1)]
        static = np.array(t["static_map"])
        k = t["num_dyn"]
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(np.where(static == 1, 1.0, 0.0), cmap="gray_r",
                      vmin=0, vmax=1, origin="upper",
                      interpolation="nearest")
            # explicit gridlines (minor-grid rendering aliases away at
            # column scale, which hid some horizontal lines)
            for g in np.arange(-0.5, M + 0.5, 1):
                ax.axhline(g, color=GRID_COLOR, linewidth=0.7, zorder=1)
                ax.axvline(g, color=GRID_COLOR, linewidth=0.7, zorder=1)
            ax.set_xlim(-0.5, M - 0.5)
            ax.set_ylim(M - 0.5, -0.5)
            ax.tick_params(which="both", bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_color("#555555")
        ax = axes[0, col]
        ax.set_title(f"Map 0{col + 1}  ($k={k}$)", fontsize=11)
        for i, p in enumerate(t["starts"]):
            ax.add_patch(plt.Rectangle((p[1] - 0.5, p[0] - 0.5), 1, 1,
                                       color=ROBOT_COLORS[i]))
        for p in t["dyn_starts"]:
            ax.add_patch(plt.Rectangle((p[1] - 0.5, p[0] - 0.5), 1, 1,
                                       color=DYN_COLOR))
        ax = axes[1, col]
        for path in t["dyn_paths"]:
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
