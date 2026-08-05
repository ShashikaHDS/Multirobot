"""Evaluate trained PPO runs against the A* baseline (all 3 heuristics).

Protocol (matches the manuscript's claims exactly):
  * 20 held-out evaluation maps per configuration (eval seeds 10000+i),
    shared between PPO and every A* heuristic -> paired per-map design.
  * PPO rolled out deterministically (model.predict(deterministic=True)).
  * Only COMPLETED training runs are evaluated (model.zip present, which
    train_paper.py writes only after learn() finishes); crashed/partial
    runs are reported and skipped.  Env parameters (threshold, max_steps,
    lidar radius) are read back from each run's config.json, not guessed.
  * Metrics per episode: success, steps, total fleet distance, max
    per-robot distance -- counted identically for both methods by the
    environment itself (realized moves only).
  * A* is deterministic given the map seed, so each (config, heuristic,
    map) is run exactly ONCE and recorded with train_seed = -1.
  * Statistics are computed on the per-map level to avoid
    pseudo-replication: the PPO value for a map is the mean over training
    seeds (n = 20 paired values per config).
      - paired Wilcoxon signed-rank vs the strongest A* heuristic on
        steps and total distance, over maps where A* succeeded and every
        PPO seed succeeded          -> results/wilcoxon.csv
      - exact McNemar (binomial) success comparison per training seed
        (20 binary pairs each)      -> results/success_tests.csv
  * summary.csv reports, for both methods, mean +/- std over the SAME
    unit (the 20 maps; PPO values are seed-averaged per map first), plus
    per-seed PPO rows for transparency.

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
    """Completed runs only: model.zip is written after learn() finishes.

    We deliberately evaluate the FINAL model, not best_model.zip: SB3's
    EvalCallback selects by mean shaped reward, and empirically the
    reward-argmax checkpoint exploits the shaping terms without reaching
    the goal (0% success while the final model reaches far higher) --
    reward-based checkpoint selection is adversarial under this reward.
    """
    runs, skipped = [], []
    for cfg_dir in sorted((base / tag).glob("N*_M*")):
        for seed_dir in sorted(cfg_dir.glob("seed*")):
            final = seed_dir / "model.zip"
            cfg_json = seed_dir / "config.json"
            if not final.exists() or not cfg_json.exists():
                skipped.append(str(seed_dir))
                continue
            n = int(cfg_dir.name.split("_")[0][1:])
            m = int(cfg_dir.name.split("_")[1][1:])
            runs.append({"n": n, "m": m, "seed": int(seed_dir.name[4:]),
                         "model": final,
                         "config": json.loads(cfg_json.read_text())})
    return runs, skipped


def env_from_run(run, override_max_steps=None):
    ec = run["config"].get("env_config", {})
    cfg = EnvConfig(num_robots=run["n"], rows=run["m"], cols=run["m"],
                    threshold_area=ec.get("threshold_area", 16),
                    max_steps=override_max_steps or ec.get("max_steps", 300),
                    lidar_radius=ec.get("lidar_radius", 1))
    return RendezvousEnv(cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="primary")
    p.add_argument("--logdir", type=str, default="runs_paper")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", type=str, default="results")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO

    runs, skipped = discover_runs(here / args.logdir, args.tag)
    for s in skipped:
        print(f"!! skipping incomplete run (no model.zip): {s}")
    if not runs:
        raise SystemExit(f"no completed runs under {here/args.logdir/args.tag}")
    print(f"found {len(runs)} completed runs")

    rows = []
    astar_done = set()
    for run in runs:
        env = env_from_run(run)
        model = PPO.load(str(run["model"]), device="cpu")
        for ep in range(args.episodes):
            seed = EVAL_SEED_BASE + ep
            met = rollout_ppo(model, env, seed)
            rows.append({"method": "ppo", "heuristic": "",
                         "n": run["n"], "m": run["m"],
                         "train_seed": run["seed"], "episode": ep, **met})
            for h in HEURISTICS:
                key = (run["m"], run["n"], h, ep)
                if key in astar_done:
                    continue
                astar_done.add(key)
                amet = run_astar_episode(env, h, seed=seed)
                rows.append({"method": "astar", "heuristic": h,
                             "n": run["n"], "m": run["m"],
                             "train_seed": -1, "episode": ep,
                             **amet, "wall_ms_per_step": ""})
        env.close()
        print(f"  evaluated N{run['n']}_M{run['m']} seed{run['seed']}")

    fields = ["method", "heuristic", "n", "m", "train_seed", "episode",
              "success", "steps", "total_distance", "max_distance",
              "wall_ms_per_step"]
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ------------------------------ summary ---------------------------- #
    def agg(values_by_key):
        """values_by_key: list of per-map metric dicts (same unit)."""
        succ = [v["success"] for v in values_by_key]
        st = [v["steps"] for v in values_by_key]
        td = [v["total_distance"] for v in values_by_key]
        md = [v["max_distance"] for v in values_by_key]
        return {"n_maps": len(values_by_key),
                "success_rate": float(np.mean(succ)),
                "steps_mean": float(np.mean(st)), "steps_std": float(np.std(st)),
                "total_dist_mean": float(np.mean(td)),
                "total_dist_std": float(np.std(td)),
                "max_dist_mean": float(np.mean(md)),
                "max_dist_std": float(np.std(md))}

    configs = sorted({(r["n"], r["m"]) for r in rows})
    summary, best_h = [], {}
    ppo_per_map = {}          # (n,m) -> {ep: seed-averaged metrics}
    for (n, m) in configs:
        ppo = [r for r in rows if r["method"] == "ppo" and r["n"] == n and r["m"] == m]
        seeds = sorted({r["train_seed"] for r in ppo})
        # per-seed rows (transparency)
        for s in seeds:
            sel = [r for r in ppo if r["train_seed"] == s]
            summary.append({"config": f"N{n}_M{m}", "method": "ppo",
                            "heuristic": "", "train_seed": s, **agg(sel)})
        # seed-averaged per-map PPO values: one value per map, same unit as A*
        per_map = {}
        for ep in sorted({r["episode"] for r in ppo}):
            sel = [r for r in ppo if r["episode"] == ep]
            per_map[ep] = {
                "success": float(np.mean([r["success"] for r in sel])),
                "all_succeed": all(r["success"] for r in sel),
                "steps": float(np.mean([r["steps"] for r in sel])),
                "total_distance": float(np.mean([r["total_distance"] for r in sel])),
                "max_distance": float(np.mean([r["max_distance"] for r in sel]))}
        ppo_per_map[(n, m)] = per_map
        summary.append({"config": f"N{n}_M{m}", "method": "ppo",
                        "heuristic": "", "train_seed": "mean",
                        **agg(list(per_map.values()))})
        h_stats = {}
        for h in HEURISTICS:
            sel = [r for r in rows if r["method"] == "astar"
                   and r["heuristic"] == h and r["n"] == n and r["m"] == m]
            h_stats[h] = agg(sel)
            summary.append({"config": f"N{n}_M{m}", "method": "astar",
                            "heuristic": h, "train_seed": -1, **h_stats[h]})
        best_h[(n, m)] = max(
            h_stats, key=lambda h: (h_stats[h]["success_rate"],
                                    -h_stats[h]["total_dist_mean"]))

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ------------------ Wilcoxon (per-map pairs, n<=20) ----------------- #
    from scipy.stats import wilcoxon, binomtest
    wrows = []
    for (n, m) in configs:
        h = best_h[(n, m)]
        astar_by_ep = {r["episode"]: r for r in rows
                       if r["method"] == "astar" and r["heuristic"] == h
                       and r["n"] == n and r["m"] == m}
        for metric in ("steps", "total_distance"):
            pairs = []
            for ep, pv in ppo_per_map[(n, m)].items():
                av = astar_by_ep.get(ep)
                if av and pv["all_succeed"] and av["success"]:
                    pairs.append((pv[metric], float(av[metric])))
            if len(pairs) >= 5:
                x = np.array([a for a, _ in pairs])
                y = np.array([b for _, b in pairs])
                if np.allclose(x, y):
                    stat, pv_ = 0.0, 1.0
                else:
                    stat, pv_ = wilcoxon(x, y)
                wrows.append({"config": f"N{n}_M{m}", "vs": f"astar_{h}",
                              "metric": metric, "n_pairs": len(pairs),
                              "ppo_mean": float(x.mean()),
                              "astar_mean": float(y.mean()),
                              "wilcoxon_stat": float(stat), "p_value": float(pv_)})
            else:
                wrows.append({"config": f"N{n}_M{m}", "vs": f"astar_{h}",
                              "metric": metric, "n_pairs": len(pairs),
                              "ppo_mean": "", "astar_mean": "",
                              "wilcoxon_stat": "", "p_value": ""})
    with open(out_dir / "wilcoxon.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wrows[0].keys()))
        w.writeheader()
        w.writerows(wrows)

    # -------- exact McNemar success comparison, per training seed -------- #
    srows = []
    for (n, m) in configs:
        h = best_h[(n, m)]
        astar_by_ep = {r["episode"]: r for r in rows
                       if r["method"] == "astar" and r["heuristic"] == h
                       and r["n"] == n and r["m"] == m}
        for s in sorted({r["train_seed"] for r in rows
                         if r["method"] == "ppo" and r["n"] == n and r["m"] == m}):
            n01 = n10 = both = neither = 0
            for r in rows:
                if r["method"] != "ppo" or r["n"] != n or r["m"] != m \
                        or r["train_seed"] != s:
                    continue
                a = astar_by_ep.get(r["episode"])
                if a is None:
                    continue
                ps, as_ = bool(r["success"]), bool(a["success"])
                both += ps and as_
                neither += (not ps) and (not as_)
                n01 += ps and not as_
                n10 += as_ and not ps
            disc = n01 + n10
            pv_ = binomtest(min(n01, n10), disc, 0.5).pvalue if disc > 0 else 1.0
            srows.append({"config": f"N{n}_M{m}", "train_seed": s,
                          "vs": f"astar_{h}", "both_succeed": both,
                          "ppo_only": n01, "astar_only": n10,
                          "neither": neither, "mcnemar_p": float(pv_)})
    with open(out_dir / "success_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)

    (out_dir / "eval_meta.json").write_text(json.dumps(
        {"episodes": args.episodes, "eval_seed_base": EVAL_SEED_BASE,
         "skipped_incomplete_runs": skipped,
         "best_heuristic_per_config":
             {f"N{n}_M{m}": best_h[(n, m)] for (n, m) in configs},
         "timestamp": time.time()}, indent=2))

    print(f"\nwrote {out_dir/'results.csv'}, summary.csv, wilcoxon.csv, "
          f"success_tests.csv")
    for s in summary:
        if s["train_seed"] in ("mean", -1):
            label = s["method"] + (f"[{s['heuristic']}]" if s["heuristic"] else "")
            print(f"  {s['config']:>8} {label:>18} "
                  f"success={s['success_rate']:.2f} "
                  f"dist={s['total_dist_mean']:.1f}")


if __name__ == "__main__":
    main()
