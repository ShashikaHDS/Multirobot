"""Evaluate trained PPO runs against the A* baseline (all 3 heuristics).

Protocol (matches the manuscript's claims exactly):
  * 20 held-out evaluation maps per configuration (eval seeds 10000+i),
    shared between PPO and every A* heuristic -> paired comparison.
  * PPO rolled out deterministically (model.predict(deterministic=True)).
  * Metrics per episode: success, steps, total fleet distance, max
    per-robot distance -- all counted identically for both methods by the
    environment itself (realized moves only).
  * Per-episode rows written to results/results.csv; summary with
    mean +/- std per (method, config) to results/summary.csv.
  * Paired Wilcoxon signed-rank (scipy) PPO vs the strongest A* heuristic
    per config, on steps and total distance over jointly successful maps,
    plus exact McNemar-style success counts, written to
    results/wilcoxon.csv.

Usage:
    python eval_paper.py --tag primary --episodes 20
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from env_paper import RendezvousEnv, EnvConfig
from astar_paper import run_astar_episode, HEURISTICS

EVAL_SEED_BASE = 10_000


def rollout_ppo(model, env: RendezvousEnv, seed: int):
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    info = {}
    t0 = time.perf_counter()
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
    wall = time.perf_counter() - t0
    return {
        "success": bool(info.get("is_success", False)),
        "steps": env.step_count,
        "total_distance": info.get("total_distance", 0),
        "max_distance": info.get("max_distance", 0),
        "wall_ms_per_step": 1000.0 * wall / max(env.step_count, 1),
    }


def discover_runs(base: Path, tag: str):
    runs = []
    for cfg_dir in sorted((base / tag).glob("N*_M*")):
        for seed_dir in sorted(cfg_dir.glob("seed*")):
            model = seed_dir / "best_model.zip"
            if not model.exists():
                model = seed_dir / "model.zip"
            if model.exists():
                n = int(cfg_dir.name.split("_")[0][1:])
                m = int(cfg_dir.name.split("_")[1][1:])
                runs.append({"n": n, "m": m,
                             "seed": int(seed_dir.name[4:]),
                             "model": model, "dir": seed_dir})
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="primary")
    p.add_argument("--logdir", type=str, default="runs_paper")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO

    runs = discover_runs(here / args.logdir, args.tag)
    if not runs:
        raise SystemExit(f"no trained runs found under {here / args.logdir / args.tag}")
    print(f"found {len(runs)} trained runs")

    rows = []
    astar_cache = {}          # (m, n, heuristic, ep) -> metrics; A* is
                              # deterministic given the seed, so run once
    for run in runs:
        cfg = EnvConfig(num_robots=run["n"], rows=run["m"], cols=run["m"],
                        max_steps=args.max_steps)
        env = RendezvousEnv(cfg)
        model = PPO.load(str(run["model"]), device="cpu")
        for ep in range(args.episodes):
            seed = EVAL_SEED_BASE + ep
            met = rollout_ppo(model, env, seed)
            rows.append({"method": "ppo", "heuristic": "",
                         "n": run["n"], "m": run["m"],
                         "train_seed": run["seed"], "episode": ep, **met})
            for h in HEURISTICS:
                key = (run["m"], run["n"], h, ep)
                if key not in astar_cache:
                    astar_cache[key] = run_astar_episode(env, h, seed=seed)
                amet = astar_cache[key]
                rows.append({"method": "astar", "heuristic": h,
                             "n": run["n"], "m": run["m"],
                             "train_seed": run["seed"], "episode": ep,
                             **amet, "wall_ms_per_step": ""})
        env.close()
        print(f"  evaluated N{run['n']}_M{run['m']} seed{run['seed']}")

    # ------------------------- write per-episode CSV ------------------- #
    fields = ["method", "heuristic", "n", "m", "train_seed", "episode",
              "success", "steps", "total_distance", "max_distance",
              "wall_ms_per_step"]
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ------------------------------ summary ---------------------------- #
    def agg(sel):
        succ = [r["success"] for r in sel]
        st = [r["steps"] for r in sel]
        td = [r["total_distance"] for r in sel]
        md = [r["max_distance"] for r in sel]
        return {"n_episodes": len(sel),
                "success_rate": np.mean(succ) if sel else 0.0,
                "steps_mean": np.mean(st), "steps_std": np.std(st),
                "total_dist_mean": np.mean(td), "total_dist_std": np.std(td),
                "max_dist_mean": np.mean(md), "max_dist_std": np.std(md)}

    configs = sorted({(r["n"], r["m"]) for r in rows})
    summary = []
    best_h = {}
    for (n, m) in configs:
        ppo = [r for r in rows if r["method"] == "ppo"
               and r["n"] == n and r["m"] == m]
        summary.append({"config": f"N{n}_M{m}", "method": "ppo",
                        "heuristic": "", **agg(ppo)})
        h_stats = {}
        for h in HEURISTICS:
            sel = [r for r in rows if r["method"] == "astar"
                   and r["heuristic"] == h and r["n"] == n and r["m"] == m
                   and r["train_seed"] == min(x["train_seed"] for x in rows
                                              if x["n"] == n and x["m"] == m)]
            h_stats[h] = agg(sel)
            summary.append({"config": f"N{n}_M{m}", "method": "astar",
                            "heuristic": h, **h_stats[h]})
        # strongest heuristic = highest success, ties by lower total distance
        best_h[(n, m)] = max(
            h_stats, key=lambda h: (h_stats[h]["success_rate"],
                                    -h_stats[h]["total_dist_mean"]))

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ------------------------------ Wilcoxon ---------------------------- #
    from scipy.stats import wilcoxon
    wrows = []
    for (n, m) in configs:
        h = best_h[(n, m)]
        for metric in ("steps", "total_distance"):
            # pair per (train_seed, episode): PPO vs best A* heuristic
            pairs = []
            for r in rows:
                if r["method"] != "ppo" or r["n"] != n or r["m"] != m:
                    continue
                mate = next((a for a in rows if a["method"] == "astar"
                             and a["heuristic"] == h and a["n"] == n
                             and a["m"] == m and a["episode"] == r["episode"]
                             and a["train_seed"] == r["train_seed"]), None)
                if mate and r["success"] and mate["success"]:
                    pairs.append((r[metric], mate[metric]))
            if len(pairs) >= 5:
                x = np.array([a for a, _ in pairs], dtype=float)
                y = np.array([b for _, b in pairs], dtype=float)
                if np.allclose(x, y):
                    stat, pv = 0.0, 1.0
                else:
                    stat, pv = wilcoxon(x, y)
                wrows.append({"config": f"N{n}_M{m}", "vs": f"astar_{h}",
                              "metric": metric, "n_pairs": len(pairs),
                              "ppo_mean": x.mean(), "astar_mean": y.mean(),
                              "wilcoxon_stat": stat, "p_value": pv})
            else:
                wrows.append({"config": f"N{n}_M{m}", "vs": f"astar_{h}",
                              "metric": metric, "n_pairs": len(pairs),
                              "ppo_mean": "", "astar_mean": "",
                              "wilcoxon_stat": "", "p_value": ""})
    with open(out_dir / "wilcoxon.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wrows[0].keys()))
        w.writeheader()
        w.writerows(wrows)

    (out_dir / "eval_meta.json").write_text(json.dumps(
        {"episodes": args.episodes, "eval_seed_base": EVAL_SEED_BASE,
         "best_heuristic_per_config":
             {f"N{n}_M{m}": best_h[(n, m)] for (n, m) in configs},
         "timestamp": time.time()}, indent=2))

    print(f"\nwrote {out_dir/'results.csv'}, summary.csv, wilcoxon.csv")
    for s in summary:
        if s["method"] == "ppo" or s["heuristic"] == "":
            print(f"  {s['config']:>8} {s['method']:>5} "
                  f"success={s['success_rate']:.2f} "
                  f"dist={s['total_dist_mean']:.1f}")


if __name__ == "__main__":
    main()
