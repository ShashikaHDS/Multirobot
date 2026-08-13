"""Energy analysis (paper Energy subsection): per-robot mission energy for
the learned policy vs the classical pipeline, under the platform energy
model  E_i = P_hotel*T + k_move*d_i .

No hardware numbers are required: energies are computed in normalised
units parameterised by the hotel-load share
    rho = hotel energy / total energy reference,   E_i(rho) = rho*T + (1-rho)*d_i,
so the figure shows the balance property across the entire plausible
power-model range. When the platform measurement is available, pass
--p-hotel/--p-drive/--step-s (and --cell-m) and a Joules table is emitted
from the same rollouts.

Rollouts are re-generated exactly as in eval_paper.py (same map seeds,
same per-sample torch seeding), so results are reproducible and
consistent with results_final2b.

    python energy_analysis.py --tag final2
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

from env_paper import RendezvousEnv, EnvConfig
from astar_paper import run_astar_episode

HUES = {2: "#8250c4", 3: "#2a78d6", 4: "#eb6834", 5: "#1baf7a"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
EVAL_SEED_BASE = 10_000
RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
PANEL_B_HEURISTICS = ("median", "minimax")


def energy_stats(T, d, rho):
    """Worst-robot share and Jain index of per-robot energy at hotel share rho."""
    e = rho * float(T) + (1.0 - rho) * np.asarray(d, dtype=float)
    if e.sum() <= 0:
        return 1.0, 1.0
    share = float(e.max() / e.mean())
    jain = float((e.sum() ** 2) / (len(e) * (e ** 2).sum()))
    return share, jain


def rollout_ppo(model, env, seed, samples):
    import torch
    out = []
    for k in range(samples):
        torch.manual_seed((seed * 1000 + k) % (2 ** 31))
        obs, _ = env.reset(seed=seed)
        term = trunc = False
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, info = env.step(a)
        out.append((env.step_count, [int(v) for v in env.distances]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="final2")
    p.add_argument("--logdir", type=str, default="runs_paper")
    p.add_argument("--best-meta", type=str,
                   default="results_final2b/eval_meta.json")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--out", type=str, default="results_energy")
    p.add_argument("--fig", type=str, default="figures/energy")
    # Joules hook (all four needed): watts idle-on, watts driving,
    # seconds per grid step, metres per cell (reporting only)
    p.add_argument("--p-hotel", type=float, default=None)
    p.add_argument("--p-drive", type=float, default=None)
    p.add_argument("--step-s", type=float, default=None)
    p.add_argument("--cell-m", type=float, default=0.17)
    p.add_argument("--from-csv", action="store_true",
                   help="replot/re-stat from an existing energy.csv "
                        "instead of re-rolling episodes")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    best_h = json.loads((here / args.best_meta).read_text())[
        "best_heuristic_per_config"]

    if args.from_csv:
        rows = []
        with open(out_dir / "energy.csv") as f:
            for r in csv.DictReader(f):
                rows.append({"config": r["config"], "n": int(r["n"]),
                             "m": int(r["m"]), "method": r["method"],
                             "heuristic": r["heuristic"],
                             "train_seed": int(r["train_seed"]),
                             "episode": int(r["episode"]),
                             "sample": int(r["sample"]),
                             "steps": int(r["steps"]),
                             "d": json.loads(r["per_robot_distances"])})
        return finish(args, here, out_dir, best_h, rows)

    from stable_baselines3 import PPO

    rows = []
    configs = []
    for cfg_dir in sorted((here / args.logdir / args.tag).glob("N*_M*")):
        n = int(cfg_dir.name.split("_")[0][1:])
        m = int(cfg_dir.name.split("_")[1][1:])
        configs.append((n, m))
        env = RendezvousEnv(EnvConfig(num_robots=n, rows=m, cols=m))

        for seed_dir in sorted(cfg_dir.glob("seed*")):
            model_p = seed_dir / "model.zip"
            if not model_p.exists():
                continue
            model = PPO.load(str(model_p), device="cpu")
            for ep in range(args.episodes):
                for si, (T, d) in enumerate(
                        rollout_ppo(model, env,
                                    EVAL_SEED_BASE + ep, args.samples)):
                    rows.append({"config": cfg_dir.name, "n": n, "m": m,
                                 "method": "ppo", "heuristic": "",
                                 "train_seed": int(seed_dir.name[4:]),
                                 "episode": ep, "sample": si,
                                 "steps": T, "d": d})

        heuristics = sorted(set(PANEL_B_HEURISTICS)
                            | {best_h[cfg_dir.name]})
        for h in heuristics:
            for ep in range(args.episodes):
                met = run_astar_episode(env, h, seed=EVAL_SEED_BASE + ep)
                rows.append({"config": cfg_dir.name, "n": n, "m": m,
                             "method": "astar", "heuristic": h,
                             "train_seed": -1, "episode": ep, "sample": 0,
                             "steps": met["steps"],
                             "d": met["per_robot_distances"]})
        env.close()
        print(f"  rolled {cfg_dir.name}", flush=True)

    return finish(args, here, out_dir, best_h, rows)


def finish(args, here, out_dir, best_h, rows):
    # ------------------------------ CSV -------------------------------- #
    fields = (["config", "n", "m", "method", "heuristic", "train_seed",
               "episode", "sample", "steps", "per_robot_distances"]
              + [f"worst_share_rho{r:g}" for r in RHO_GRID]
              + [f"jainE_rho{r:g}" for r in RHO_GRID])
    with open(out_dir / "energy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rec = {k: r[k] for k in fields[:9] if k in r}
            rec["per_robot_distances"] = json.dumps(r["d"])
            for rho in RHO_GRID:
                s, j = energy_stats(r["steps"], r["d"], rho)
                rec[f"worst_share_rho{rho:g}"] = round(s, 4)
                rec[f"jainE_rho{rho:g}"] = round(j, 4)
            w.writerow(rec)

    # -------------------- aggregates for the figure -------------------- #
    def series(config, method, heuristic, rho):
        shares = [energy_stats(r["steps"], r["d"], rho)[0] for r in rows
                  if r["config"] == config and r["method"] == method
                  and (method == "ppo" or r["heuristic"] == heuristic)]
        return float(np.mean(shares))

    rho_axis = np.linspace(0, 1, 21)
    panel_a_configs = [c for c in sorted(set(r["config"] for r in rows))
                       if c.endswith("_M20") and not c.startswith("N2")]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": TEXT, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 2.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    for config in panel_a_configs:
        nrob = int(config.split("_")[0][1:])
        color = HUES[nrob]
        h = best_h[config]
        ya = [series(config, "ppo", "", r) for r in rho_axis]
        yb = [series(config, "astar", h, r) for r in rho_axis]
        axA.plot(rho_axis, ya, color=color, linewidth=1.7,
                 label=f"PPO  N={nrob}")
        axA.plot(rho_axis, yb, color=color, linewidth=1.4,
                 linestyle=(0, (3, 2)), label=f"A*  N={nrob}")
    axA.set_xlabel(r"Hotel-load share  $\rho$")
    axA.set_ylabel("Worst-robot energy / fleet mean")
    axA.axhline(1.0, color=MUTED, linewidth=0.8)
    axA.grid(True, color=GRID, linewidth=0.6)
    axA.set_axisbelow(True)
    axA.legend(frameon=False, ncol=2, handlelength=1.6)

    # panel B: energy-Jain at rho=0.5 per config, PPO vs median vs minimax
    all_configs = sorted(set(r["config"] for r in rows))
    xs = np.arange(len(all_configs))
    markers = {"ppo": ("o", TEXT), "median": ("s", "#b3453b"),
               "minimax": ("^", "#3b7bb3")}
    for label, (marker, color) in markers.items():
        ys = []
        for config in all_configs:
            sel = [energy_stats(r["steps"], r["d"], 0.5)[1] for r in rows
                   if r["config"] == config
                   and ((label == "ppo" and r["method"] == "ppo")
                        or (label != "ppo" and r["heuristic"] == label))]
            ys.append(float(np.mean(sel)) if sel else np.nan)
        axB.plot(xs, ys, marker=marker, color=color, linewidth=1.2,
                 markersize=5, label={"ppo": "PPO", "median": "A* median",
                                      "minimax": "A* minimax"}[label],
                 markeredgecolor="white", markeredgewidth=0.5)
    axB.set_xticks(xs)
    axB.set_xticklabels([c.replace("_", "\n") for c in all_configs],
                        fontsize=6.5)
    axB.set_ylabel(r"Energy Jain index  ($\rho=0.5$)")
    axB.set_ylim(0.94, 1.003)
    axB.grid(True, color=GRID, linewidth=0.6)
    axB.set_axisbelow(True)
    axB.legend(frameon=False)
    for ax in (axA, axB):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    figp = here / args.fig
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figp) + ".pdf", bbox_inches="tight")
    fig.savefig(str(figp) + ".png", dpi=300, bbox_inches="tight")
    print(f"wrote {figp}.pdf / .png and {out_dir/'energy.csv'}")

    # ------------------- Wilcoxon on energy-Jain (rho=0.5) ------------- #
    from scipy.stats import wilcoxon
    stats = {}
    for config in all_configs:
        ppo_map = defaultdict(list)
        for r in rows:
            if r["config"] == config and r["method"] == "ppo":
                ppo_map[r["episode"]].append(
                    energy_stats(r["steps"], r["d"], 0.5)[1])
        for h in PANEL_B_HEURISTICS:
            a_map = {r["episode"]: energy_stats(r["steps"], r["d"], 0.5)[1]
                     for r in rows if r["config"] == config
                     and r["heuristic"] == h}
            x = np.array([np.mean(v) for k, v in sorted(ppo_map.items())
                          if k in a_map])
            y = np.array([a_map[k] for k in sorted(ppo_map) if k in a_map])
            if len(x) >= 5 and not np.allclose(x, y):
                stat, pv = wilcoxon(x, y)
            else:
                stat, pv = 0.0, 1.0
            stats[f"{config}_vs_{h}"] = {
                "ppo_mean": round(float(x.mean()), 4),
                "astar_mean": round(float(y.mean()), 4),
                "p_value": float(pv)}
            print(f"  {config} energy-Jain PPO {x.mean():.3f} vs "
                  f"A*[{h}] {y.mean():.3f}  p={pv:.2g}")

    # ------------------------- optional Joules ------------------------- #
    joules = None
    if args.p_hotel and args.p_drive and args.step_s:
        joules = {}
        for config in all_configs:
            for method, h in ([("ppo", "")]
                              + [("astar", best_h[config])]):
                sel = [r for r in rows if r["config"] == config
                       and r["method"] == method
                       and (method == "ppo" or r["heuristic"] == h)]
                e = [args.p_hotel * r["steps"] * args.step_s
                     + (args.p_drive - args.p_hotel) * args.step_s
                     * float(np.sum(r["d"])) for r in sel]
                joules[f"{config}_{method}"] = round(float(np.mean(e)), 1)
                print(f"  {config} {method}: mean fleet energy "
                      f"{np.mean(e):.1f} J")

    (out_dir / "energy_meta.json").write_text(json.dumps(
        {"tag": args.tag, "episodes": args.episodes,
         "samples": args.samples, "rho_grid": RHO_GRID,
         "best_heuristic_per_config": best_h,
         "wilcoxon_energy_jain_rho0.5": stats,
         "joules": joules,
         "cell_m": args.cell_m}, indent=2))


if __name__ == "__main__":
    main()
