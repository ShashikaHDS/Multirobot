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

Easiest path, `run_smoke.sh` finds the Isaac Sim python itself (an
importable `isaacsim` in the active env, else the usual `python.sh`
locations), installs SB3 if missing, and runs the protocol:

```bash
cd <repo>/paper_rev2/isaac
bash run_smoke.sh          # smoke, 2 maps at N=4, ~5 min
bash run_smoke.sh full     # full protocol, ~1-3 h
bash run_smoke.sh noise    # localisation-noise sweep
```

If detection fails it prints every location it searched, so pass the
correct python explicitly using the manual commands below.

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

## Watch it live / record a video

`visualize.py` is the front end for the capture pipeline (`capture_media.py`
+ `smorphi_model.py`): it replays one seeded episode of the trained policy
through Isaac Sim with the Smorphi CAD robots, their LiDAR returns, the
explored-area overlay and breadcrumb trails, and shows it in the Isaac Sim
window while it records.

```bash
cd <repo>/paper_rev2/isaac
~/isaac-sim*/python.sh visualize.py --n 4                        # watch N=4 live and record
~/isaac-sim*/python.sh visualize.py --n 5 --m 25 --map 0         # N5_M25 on held-out map 0
~/isaac-sim*/python.sh visualize.py --n 4 --no-video --stills 1  # window + stills, no MP4
~/isaac-sim*/python.sh visualize.py --n 4 --headless             # record with no window (ssh)
```

`--n` picks the fleet size (trained final2m models: N = 2, 3, 4, 5 on
20x20 and N = 5 on 25x25), `--m` the grid (default 20), `--train-seed`
the training seed 0-2, `--map` the held-out map (generator seed
10000+map, same as the evaluation protocol), `--sample` the stochastic
rollout index, `--stills` the policy steps at which PNG stills are saved
(default `0,5,10,15,20`; the final frame is always saved), `--no-video`
skips the MP4 writers, `--headless` runs without a window.  Output goes
to `paper_rev2/media_isaac_N<n>_M<m>_map<map>/` unless `--out` is given:
`top.mp4` (1080x1080) and `persp.mp4` (1920x1080) at 30 fps in real time,
plus `still_<cam>_step<k>.png` and `still_<cam>_final.png`.

Under the hood `visualize.py` builds an argv for `capture_media.main()`
(`--config/--train-seed/--map/--sample/--still-steps/--out`, plus
`--windowed` / `--no-video`); call `capture_media.py` directly for the
remaining knobs (`--quality`, `--stride`, `--fps`, `--no-breadcrumbs`,
`--reveal`). The robot geometry comes from `smorphi_model.build_smorphi`
(the exported CAD in `ref/cad/`, converted once by `stl_to_usd.py`).
Everything drawn is visual-only: no colliders, so the PhysX raycasts that
feed the policy are the same as in the validation runs.

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
