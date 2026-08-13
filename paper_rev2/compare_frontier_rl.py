"""Focused frontier-vs-RL comparison on 8 fixed held-out maps
(6x 20x20, seeds 10000-10005; 2x 25x25, seeds 10000-10001), all fleet
sizes N=2-5 on the 20x20 maps and N=5 on the 25x25 maps.

Metrics per (map, N, method): success rate, steps to converge (censored
at the cap), fleet distance, max per-robot distance, Jain fairness of
distance, obstacle contacts, and map-revealed fraction at episode end
(information efficiency: the frontier method must explore to decide,
the policy converges on less information).

RL rows use the final2 models under the standard stochastic protocol
(3 training seeds x 5 seeded samples = 15 rollouts per cell); the
frontier method is deterministic (1 episode per danger-buffer variant,
stronger variant reported).

    python compare_frontier_rl.py [--smoke]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_paper import RendezvousEnv, EnvConfig, UNKNOWN, OBSTACLE
from frontier_baseline import run_frontier_episode

HUES = {2: "#8250c4", 3: "#2a78d6", 4: "#eb6834", 5: "#1baf7a"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
FRONTIER_C, PPO_C = "#b3453b", "#0b0b0b"

MAPS_20 = [10000 + i for i in range(6)]
MAPS_25 = [10000 + i for i in range(2)]
CELLS = ([(s, 20, n) for s in MAPS_20 for n in (2, 3, 4, 5)]
         + [(s, 25, 5) for s in MAPS_25])


def jain(d):
    d = np.asarray(d, dtype=float)
    return float((d.sum() ** 2) / (len(d) * (d ** 2).sum())) if d.sum() > 0 else 1.0


def revealed_frac(env):
    return float(np.count_nonzero(env.known_map != UNKNOWN)
                 / env.known_map.size)


def rollout_ppo(model, env, seed, k):
    import torch
    torch.manual_seed((seed * 1000 + k) % (2 ** 31))
    obs, _ = env.reset(seed=seed)
    term = trunc = False
    info = {}
    n_obs = 0
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=False)
        obs, r, term, trunc, info = env.step(a)
        n_obs += info["obstacle_collisions"]
    return {"success": bool(info.get("is_success")),
            "steps": env.step_count,
            "total_distance": int(env.distances.sum()),
            "max_distance": int(env.distances.max()),
            "jain": jain(env.distances),
            "obs_collisions": n_obs,
            "revealed": revealed_frac(env),
            "per_robot_distances": [int(v) for v in env.distances]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="one cell only (map 10000, N=4, 20x20)")
    p.add_argument("--out", type=str, default="results_frontier_rl")
    p.add_argument("--from-csv", action="store_true",
                   help="replot from existing results.csv (no rollouts)")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        def num(v):
            if v == "True":
                return 1.0
            if v == "False":
                return 0.0
            return float(v)
        rows = []
        with open(out_dir / "results.csv") as f:
            for r in csv.DictReader(f):
                rows.append({"map_seed": int(r["map_seed"]),
                             "m": int(r["m"]), "n": int(r["n"]),
                             "method": r["method"], "variant": r["variant"],
                             "rollout": int(r["rollout"]),
                             "success": num(r["success"]),
                             "steps": int(r["steps"]),
                             "total_distance": int(r["total_distance"]),
                             "max_distance": int(r["max_distance"]),
                             "jain": float(r["jain"]),
                             "obs_collisions": float(r["obs_collisions"]),
                             "revealed": float(r["revealed"]),
                             "per_robot_distances": r["per_robot_distances"]})
        return postprocess(args, here, out_dir, rows)

    from stable_baselines3 import PPO

    cells = [(10000, 20, 4)] if args.smoke else CELLS
    rows = []
    model_cache = {}
    env_cache = {}

    for (seed, m, n) in cells:
        if (n, m) not in env_cache:
            env_cache[(n, m)] = RendezvousEnv(
                EnvConfig(num_robots=n, rows=m, cols=m))
        env = env_cache[(n, m)]

        # ---- PPO: 3 training seeds x 5 samples ----
        for ts in (0, 1, 2):
            key = (n, m, ts)
            if key not in model_cache:
                mp = here / "runs_paper" / "final2" / f"N{n}_M{m}" \
                    / f"seed{ts}" / "model.zip"
                model_cache[key] = PPO.load(str(mp), device="cpu")
            for k in range(5):
                met = rollout_ppo(model_cache[key], env, seed, k)
                met["per_robot_distances"] = json.dumps(
                    met["per_robot_distances"])
                rows.append({"map_seed": seed, "m": m, "n": n,
                             "method": "ppo", "variant": f"seed{ts}",
                             "rollout": k, **met})

        # ---- frontier: both variants, deterministic ----
        for variant, ud in (("danger", True), ("no_danger", False)):
            met = dict(run_frontier_episode(env, seed=seed, use_danger=ud))
            met["revealed"] = revealed_frac(env)
            met.pop("cf_success", None)
            met.pop("robot_collisions", None)
            met["per_robot_distances"] = json.dumps(
                met["per_robot_distances"])
            rows.append({"map_seed": seed, "m": m, "n": n,
                         "method": "frontier", "variant": variant,
                         "rollout": 0, **met})
        print(f"  cell map{seed} M{m} N{n} done", flush=True)

    for env in env_cache.values():
        env.close()

    fields = ["map_seed", "m", "n", "method", "variant", "rollout",
              "success", "steps", "total_distance", "max_distance",
              "jain", "obs_collisions", "revealed", "per_robot_distances"]
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if args.smoke:
        for r in rows:
            print({k: r[k] for k in fields[:12]})
        return
    return postprocess(args, here, out_dir, rows)


def postprocess(args, here, out_dir, rows):
    # ------------------- aggregation helpers -------------------------- #
    def ppo_cell(seed, m, n):
        sel = [r for r in rows if r["method"] == "ppo"
               and r["map_seed"] == seed and r["m"] == m and r["n"] == n]
        return {k: float(np.mean([r[k] for r in sel]))
                for k in ("success", "steps", "total_distance",
                          "max_distance", "jain", "obs_collisions",
                          "revealed")}

    def frontier_cell(seed, m, n):
        """Stronger danger-variant per cell (higher success, then fewer steps)."""
        best = None
        for r in rows:
            if r["method"] == "frontier" and r["map_seed"] == seed \
                    and r["m"] == m and r["n"] == n:
                cand = {k: float(r[k]) for k in
                        ("success", "steps", "total_distance",
                         "max_distance", "jain", "obs_collisions",
                         "revealed")}
                cand["variant"] = r["variant"]
                if best is None or (cand["success"], -cand["steps"]) > \
                        (best["success"], -best["steps"]):
                    best = cand
        return best

    metrics = ["success", "steps", "total_distance", "max_distance",
               "jain", "obs_collisions", "revealed"]

    # ---------------------- summary per (N, M) ------------------------ #
    groups = [(n, 20, MAPS_20) for n in (2, 3, 4, 5)] + [(5, 25, MAPS_25)]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "method"] + metrics)
        for (n, m, seeds) in groups:
            for label, fn in (("ppo", ppo_cell), ("frontier", frontier_cell)):
                vals = [fn(s, m, n) for s in seeds]
                w.writerow([f"N{n}_M{m}", label] +
                           [round(float(np.mean([v[k] for v in vals])), 3)
                            for k in metrics])

    # ------------------- per-map table (N=4 / N=5) -------------------- #
    lines = ["| Map | Method | Succ | Steps | Dist | MaxD | Jain | Contacts | Revealed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (seed, m, n) in [(s, 20, 4) for s in MAPS_20] + \
                        [(s, 25, 5) for s in MAPS_25]:
        for label, fn in (("PPO", ppo_cell), ("Frontier", frontier_cell)):
            v = fn(seed, m, n)
            lines.append(
                f"| {m}x{m} #{seed - 10000 + 1} | {label} | "
                f"{v['success']:.2f} | {v['steps']:.0f} | "
                f"{v['total_distance']:.0f} | {v['max_distance']:.0f} | "
                f"{v['jain']:.3f} | {v['obs_collisions']:.1f} | "
                f"{v['revealed']:.2f} |")
    (out_dir / "per_map.md").write_text("\n".join(lines))

    # --------------------------- map figure --------------------------- #
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "text.color": TEXT,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.7))
    layout = [(s, 20) for s in MAPS_20] + [(s, 25) for s in MAPS_25]
    for ax, (seed, m) in zip(axes.flat, layout):
        n_show = 4 if m == 20 else 5
        env = RendezvousEnv(EnvConfig(num_robots=n_show, rows=m, cols=m))
        env.reset(seed=seed)
        img = np.ones((m, m, 3))
        img[env.grid_map == OBSTACLE] = (0.12, 0.12, 0.12)
        ax.imshow(img, interpolation="nearest")
        ax.scatter(env.positions[:, 1], env.positions[:, 0], s=28,
                   c=HUES[n_show], edgecolors="white", linewidths=0.6,
                   zorder=3)
        env.close()
        ax.set_title(f"{m}×{m}  map {seed - 10000 + 1}", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(here / "figures" / "frontier_rl_maps") + ext,
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------ comparison figure ----------------------- #
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.5))
    ns = [2, 3, 4, 5]
    for label, fn, color, ls in (("PPO", ppo_cell, PPO_C, "-"),
                                 ("Frontier", frontier_cell, FRONTIER_C,
                                  (0, (4, 2)))):
        succ = [np.mean([fn(s, 20, n)["success"] for s in MAPS_20])
                for n in ns]
        a1.plot(ns, succ, color=color, linestyle=ls, marker="o",
                markersize=4, linewidth=1.6, label=label,
                markeredgecolor="white", markeredgewidth=0.5)
        s25 = np.mean([fn(s, 25, 5)["success"] for s in MAPS_25])
        a1.scatter([5.25], [s25], color=color, marker="s", s=22, zorder=3)
        rev = [np.mean([fn(s, 20, n)["revealed"] for s in MAPS_20])
               for n in ns]
        a3.plot(ns, rev, color=color, linestyle=ls, marker="o",
                markersize=4, linewidth=1.6, label=label,
                markeredgecolor="white", markeredgewidth=0.5)
    a1.set_xlabel("Fleet size N (squares: 25×25)")
    a1.set_ylabel("Success rate")
    a1.set_ylim(0, 1.02)
    a1.set_xticks(ns)
    a1.legend(frameon=False)

    # per-map success bars at N=4 (20x20) / N=5 (25x25)
    labels, pv, fv = [], [], []
    for (seed, m, n) in [(s, 20, 4) for s in MAPS_20] + \
                        [(s, 25, 5) for s in MAPS_25]:
        labels.append(f"{'25:' if m == 25 else ''}#{seed - 10000 + 1}")
        pv.append(ppo_cell(seed, m, n)["success"])
        fv.append(frontier_cell(seed, m, n)["success"])
    x = np.arange(len(labels))
    a2.bar(x - 0.19, pv, width=0.36, color=PPO_C)
    a2.bar(x + 0.19, fv, width=0.36, color=FRONTIER_C)
    a2.set_xticks(x)
    a2.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    a2.set_ylabel("Success (N=4 / N=5)")
    a2.set_ylim(0, 1.02)

    a3.set_xlabel("Fleet size N")
    a3.set_ylabel("Map revealed at end")
    a3.set_ylim(0, 1.02)
    a3.set_xticks(ns)
    for ax in (a1, a2, a3):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(here / "figures" / "frontier_rl") + ext,
                    dpi=300, bbox_inches="tight")

    print(f"wrote {out_dir}/results.csv, summary.csv, per_map.md and "
          f"figures/frontier_rl(.pdf/.png), frontier_rl_maps(.pdf/.png)")


if __name__ == "__main__":
    main()
