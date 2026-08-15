# Continuous-space zero-shot validation (Isaac Sim)

Runs the **already-trained** `final2m` policies in a continuous world with
raycast LiDAR sensing and velocity-limited holonomic motion. No
retraining, no new models: this measures how much of the grid-world
performance survives continuous execution, which is the evidence a
reviewer asks for between "20x20 grid" and "two robots in a lab".

## What is here

| file | role |
|---|---|
| `bridge.py` | backend-independent core: occupancy mapping (`grid` oracle or `lidar` raycast), conflict resolution delegated to `env_paper`, the lockstep deployment loop, metrics |
| `mock_env.py` | pure-Python continuous backend (exact kinematics + segment raycasts); runs anywhere, no Isaac needed |
| `isaac_env.py` | Isaac Sim backend: scene from the grid, PhysX raycast LiDAR, kinematic robot prims |
| `run_validation.py` | CLI, writes `results.csv` / `summary.csv` / `meta.json` |
| `test_bridge.py` | unit tests + **parity test** vs `env_paper` |

## Why the parity test matters

With `--reveal grid` and no noise, the continuous pipeline is proven to
reproduce `env_paper` rollouts *step for step* (identical cell
trajectories, step counts, success). Verified locally on 3 maps at N=4.
So any drop measured under `--reveal lidar` or `--noise > 0` comes from
the physics and sensing, not from bridge bugs.

## Run it on the 5090

One-time, install SB3 into the Isaac Sim python:

```bash
~/isaacsim/python.sh -m pip install stable-baselines3 gymnasium
```

Smoke test first (2 maps, ~5 min, confirms the Isaac imports work):

```bash
cd ~/<repo>/paper_rev2/isaac
~/isaacsim/python.sh run_validation.py --backend isaac --maps 2 --n 4 \
    --m 20 --samples 2 --reveal lidar --out results_isaac_smoke
```

Full protocol (10 held-out maps x N in {3,4,5} x 5 samples x 3 training
seeds; expect roughly 1-3 h headless):

```bash
~/isaacsim/python.sh run_validation.py --backend isaac --maps 10 \
    --n 3,4,5 --m 20,25 --samples 5 --reveal lidar --out results_isaac
```

Localisation-noise sweep (sigma in metres on reported poses, cell is
0.25 m by default, so 0.10 is a hard test):

```bash
for s in 0.00 0.05 0.10; do
  ~/isaacsim/python.sh run_validation.py --backend isaac --maps 10 --n 4 \
      --m 20 --samples 5 --reveal lidar --noise $s --out results_isaac_n$s
done
```

Video capture for supplementary material (GUI, one map):

```bash
~/isaacsim/python.sh run_validation.py --backend isaac --maps 1 --n 4 \
    --m 20 --samples 1 --windowed --out results_isaac_video
```

Then push results back so they can be analysed:

```bash
git add paper_rev2/results_isaac* && git commit -m "isaac validation results" \
  && git push origin revision-2026-06
```

## If the Isaac imports fail

`isaac_env.py` tries `isaacsim.SimulationApp` then `omni.isaac.kit`, and
`isaacsim.core.api` then `omni.isaac.core`. If your build differs, send
the traceback and it will be extended. Meanwhile everything is runnable
with `--backend mock`, which exercises the identical pipeline with
analytic kinematics and raycasts:

```bash
python run_validation.py --backend mock --maps 10 --n 3,4,5 --samples 5
```

## Useful flags

`--cell-size` metres per grid cell (0.25 default; Smorphi footprint is
0.17 m). `--vmax` m/s. `--n-beams` LiDAR beams (72 default).
`--reveal grid|lidar`. `--noise` pose sigma in metres. `--maps`,
`--samples`, `--n`, `--m`, `--tag`, `--out`.

## Local reference numbers (mock backend, this laptop)

N4_M20, 4 maps x 3 samples x 3 seeds, `--reveal lidar`: success 0.97
(grid-world reference for the same configuration is 0.95).
