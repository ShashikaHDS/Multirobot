"""Paired comparison of the grid evaluation against the Isaac Sim
continuous-space validation on the same held-out maps and training seeds.

    python analyze_isaac.py [--isaac results_isaac] [--grid results_final2m]

Grid rows (results_final2m/results.csv) are per (config, train_seed,
episode) means over the 5 seeded samples; Isaac rows are per sample, so
they are aggregated the same way before pairing.  Writes
<isaac>/comparison_grid_vs_isaac.csv and prints the table plus paired
Wilcoxon tests on per-map success, steps, and Jain.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[0]


def load_grid(path, maps):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["method"] != "ppo" or int(r["episode"]) >= maps:
            continue
        key = (f"N{r['n']}_M{r['m']}", int(r["train_seed"]), int(r["episode"]))
        out[key] = {"success": float(r["success"]),
                    "steps": float(r["steps"]),
                    "dist": float(r["total_distance"]),
                    "jain": float(r["jain"])}
    return out


def load_isaac(path):
    acc = defaultdict(list)
    for r in csv.DictReader(open(path)):
        key = (r["config"], int(r["train_seed"]), int(r["episode"]))
        acc[key].append(r)
    out = {}
    for key, rs in acc.items():
        out[key] = {"success": np.mean([float(r["success"]) for r in rs]),
                    "steps": np.mean([float(r["steps"]) for r in rs]),
                    "dist": np.mean([float(r["dist_cells"]) for r in rs]),
                    "jain": np.mean([float(r["jain"]) for r in rs])}
    return out


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", default="results_isaac")
    ap.add_argument("--grid", default="results_final2m")
    args = ap.parse_args()

    isaac = load_isaac(BASE / args.isaac / "results.csv")
    maps = max(k[2] for k in isaac) + 1
    grid = load_grid(BASE / args.grid / "results.csv", maps)

    configs = sorted({k[0] for k in isaac},
                     key=lambda c: (int(c.split("_M")[1]), int(c[1])))
    rows = []
    print(f"{'config':8s} {'pairs':>5s} {'succ grid':>9s} {'succ isaac':>10s} "
          f"{'p':>7s} {'steps grid':>10s} {'steps isaac':>11s} {'p':>7s} "
          f"{'jain grid':>9s} {'jain isaac':>10s} {'p':>7s}")
    pooled_g, pooled_i = [], []
    for cfg in configs:
        keys = sorted(k for k in isaac if k[0] == cfg and k in grid)
        g = {m: np.array([grid[k][m] for k in keys])
             for m in ("success", "steps", "dist", "jain")}
        i = {m: np.array([isaac[k][m] for k in keys])
             for m in ("success", "steps", "dist", "jain")}
        pooled_g += list(g["success"])
        pooled_i += list(i["success"])
        p_s = wilcoxon_p(i["success"], g["success"])
        p_t = wilcoxon_p(i["steps"], g["steps"])
        p_j = wilcoxon_p(i["jain"], g["jain"])
        print(f"{cfg:8s} {len(keys):5d} {g['success'].mean():9.3f} "
              f"{i['success'].mean():10.3f} {p_s:7.3f} {g['steps'].mean():10.1f} "
              f"{i['steps'].mean():11.1f} {p_t:7.3f} {g['jain'].mean():9.3f} "
              f"{i['jain'].mean():10.3f} {p_j:7.3f}")
        rows.append({"config": cfg, "pairs": len(keys),
                     "success_grid": round(g["success"].mean(), 4),
                     "success_isaac": round(i["success"].mean(), 4),
                     "success_p": round(p_s, 4),
                     "steps_grid": round(g["steps"].mean(), 2),
                     "steps_isaac": round(i["steps"].mean(), 2),
                     "steps_p": round(p_t, 4),
                     "dist_grid": round(g["dist"].mean(), 2),
                     "dist_isaac": round(i["dist"].mean(), 2),
                     "jain_grid": round(g["jain"].mean(), 4),
                     "jain_isaac": round(i["jain"].mean(), 4),
                     "jain_p": round(p_j, 4)})
    pg, pi = np.array(pooled_g), np.array(pooled_i)
    print(f"\npooled over {len(pg)} (config, seed, map) pairs: grid "
          f"{pg.mean():.3f}  isaac {pi.mean():.3f}  diff {pi.mean()-pg.mean():+.3f}  "
          f"Wilcoxon p={wilcoxon_p(pi, pg):.3f}")
    better = int(np.sum(pi > pg)); worse = int(np.sum(pi < pg))
    print(f"maps where isaac > grid: {better}, isaac < grid: {worse}, "
          f"tied: {len(pg) - better - worse}")

    out = BASE / args.isaac / "comparison_grid_vs_isaac.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
