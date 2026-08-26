"""Continuous-space zero-shot validation of the trained final2m policies.

Runs the standard held-out evaluation protocol (map seeds 10000+i, seeded
stochastic rollouts, censored at the 300-step policy cap) through a
continuous backend, either the pure-Python MockBackend (any machine) or
IsaacBackend (Isaac Sim on the 5090).  Maps and start positions are
obtained from the training environment itself (an env instance acts as
the oracle), so they are bit-identical to the grid evaluation and the
paired per-map design carries over.

Examples
  python run_validation.py --backend mock --maps 3 --n 4 --samples 3
  ~/isaacsim/python.sh run_validation.py --backend isaac --maps 10 \
      --n 3,4,5 --samples 5 --reveal lidar --out results_isaac

Outputs <out>/results.csv, <out>/summary.csv, <out>/meta.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

from env_paper import RendezvousEnv, EnvConfig  # noqa: E402
from bridge import LockstepRunner, RunnerConfig  # noqa: E402

EVAL_SEED_BASE = 10_000


def _alias_numpy_core():
    """Let numpy-1.x hosts unpickle objects saved under numpy 2.

    The models were trained with numpy 2 (module path ``numpy._core``);
    Isaac Sim's bundled python ships numpy 1.x (``numpy.core``).
    """
    if hasattr(np, "_core"):
        return
    import importlib
    core = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", core)
    for sub in ("multiarray", "numeric", "umath", "_multiarray_umath",
                "fromnumeric", "numerictypes"):
        try:
            mod = importlib.import_module(f"numpy.core.{sub}")
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core.{sub}", mod)


def load_policy(path: Path, spaces_env):
    """Load a trained PPO robustly across python/numpy versions.

    Pickled schedules (py3.9 cloudpickle lambdas) and gym spaces (numpy-2
    arrays) are the fragile parts of an SB3 zip; supply them directly
    instead so only the network weights are read from the file.
    """
    from stable_baselines3 import PPO
    _alias_numpy_core()
    custom = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        "observation_space": spaces_env.observation_space,
        "action_space": spaces_env.action_space,
    }
    return PPO.load(str(path), device="cpu", custom_objects=custom)


def discover_runs(logdir: Path, tag: str, ns, ms):
    runs = []
    for cfg_dir in sorted((logdir / tag).iterdir()):
        if not cfg_dir.is_dir() or not cfg_dir.name.startswith("N"):
            continue
        n = int(cfg_dir.name.split("_")[0][1:])
        m = int(cfg_dir.name.split("_")[1][1:])
        if n not in ns or m not in ms:
            continue
        for seed_dir in sorted(cfg_dir.glob("seed*")):
            model = seed_dir / "model.zip"
            if model.exists():
                runs.append({"n": n, "m": m,
                             "train_seed": int(seed_dir.name[4:]),
                             "model": model,
                             "config": f"N{n}_M{m}"})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "isaac"], default="mock")
    ap.add_argument("--tag", default="final2m")
    ap.add_argument("--logdir", default=str(HERE.parents[0] / "runs_paper"))
    ap.add_argument("--maps", type=int, default=10)
    ap.add_argument("--n", default="3,4,5")
    ap.add_argument("--m", default="20,25")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--reveal", choices=["lidar", "grid"], default="lidar")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--cell-size", type=float, default=0.25)
    ap.add_argument("--vmax", type=float, default=0.5)
    ap.add_argument("--n-beams", type=int, default=72)
    ap.add_argument("--out", default="results_isaac")
    ap.add_argument("--windowed", action="store_true",
                    help="isaac backend with GUI (for video capture)")
    args = ap.parse_args()

    ns = {int(x) for x in args.n.split(",")}
    ms = {int(x) for x in args.m.split(",")}
    out = HERE.parents[0] / args.out
    out.mkdir(exist_ok=True)

    if args.backend == "mock":
        from mock_env import MockBackend
        backend = MockBackend(vmax=args.vmax, n_beams=args.n_beams)
    else:
        from isaac_env import IsaacBackend
        backend = IsaacBackend(vmax=args.vmax, n_beams=args.n_beams,
                               headless=not args.windowed)

    runs = discover_runs(Path(args.logdir), args.tag, ns, ms)
    if not runs:
        raise SystemExit("no runs found; check --logdir/--tag/--n/--m")
    print(f"{len(runs)} runs x {args.maps} maps x {args.samples} samples "
          f"on backend={args.backend}, reveal={args.reveal}, "
          f"noise={args.noise}")

    rows = []
    t0 = time.time()
    for run in runs:
        oracle = RendezvousEnv(EnvConfig(num_robots=run["n"],
                                         rows=run["m"], cols=run["m"]))
        model = load_policy(run["model"], oracle)
        for ep in range(args.maps):
            map_seed = EVAL_SEED_BASE + ep
            oracle.reset(seed=map_seed)
            grid = oracle.grid_map.copy()
            starts = [tuple(p) for p in oracle.positions]
            for k in range(args.samples):
                import torch
                torch.manual_seed((map_seed * 1000 + k) % (2 ** 31))
                cfg = RunnerConfig(cell_size=args.cell_size,
                                   reveal=args.reveal,
                                   noise_sigma=args.noise,
                                   n_beams=args.n_beams)
                runner = LockstepRunner(
                    model, backend, grid, starts, cfg,
                    noise_rng=np.random.default_rng(map_seed * 1000 + k))
                r = runner.run()
                rows.append({
                    "config": run["config"], "n": run["n"], "m": run["m"],
                    "train_seed": run["train_seed"], "episode": ep,
                    "sample": k, "success": int(r.success),
                    "steps": r.steps, "sim_time_s": round(r.sim_time_s, 2),
                    "dist_m": round(float(r.dist_m.sum()), 3),
                    "dist_cells": round(float(r.dist_cells.sum()), 2),
                    "max_dist_cells": round(float(r.dist_cells.max()), 2),
                    "jain": round(r.jain, 4),
                    "obstacle_rejections": r.obstacle_rejections,
                    "robot_conflicts": r.robot_conflicts,
                })
            done = len(rows)
            print(f"  {run['config']} seed{run['train_seed']} "
                  f"map {ep}: rows={done} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg = defaultdict(list)
    for r in rows:
        agg[r["config"]].append(r)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n_rollouts", "success_rate", "steps_mean",
                    "dist_cells_mean", "max_dist_cells_mean", "jain_mean",
                    "obstacle_rejections_mean", "sim_time_mean_s"])
        for cfg_name, rs in sorted(agg.items()):
            w.writerow([
                cfg_name, len(rs),
                round(float(np.mean([r["success"] for r in rs])), 4),
                round(float(np.mean([r["steps"] for r in rs])), 1),
                round(float(np.mean([r["dist_cells"] for r in rs])), 1),
                round(float(np.mean([r["max_dist_cells"] for r in rs])), 1),
                round(float(np.mean([r["jain"] for r in rs])), 4),
                round(float(np.mean([r["obstacle_rejections"] for r in rs])), 1),
                round(float(np.mean([r["sim_time_s"] for r in rs])), 1),
            ])
    with open(out / "meta.json", "w") as f:
        json.dump({"backend": args.backend, "tag": args.tag,
                   "maps": args.maps, "samples": args.samples,
                   "reveal": args.reveal, "noise": args.noise,
                   "cell_size": args.cell_size, "vmax": args.vmax,
                   "n_beams": args.n_beams,
                   "eval_seed_base": EVAL_SEED_BASE,
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=1)
    backend.close()
    print("wrote", out / "summary.csv")
    for cfg_name, rs in sorted(agg.items()):
        print(f"  {cfg_name}: success "
              f"{np.mean([r['success'] for r in rs]):.2f} over {len(rs)}")


if __name__ == "__main__":
    main()
