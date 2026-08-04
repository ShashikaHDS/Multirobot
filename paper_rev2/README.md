# paper_rev2 — canonical code for the rendezvous paper revision

Single source of truth replacing the legacy `v5.py` line. Fixes applied
relative to the legacy environment: collision penalties actually reach the
reward (accumulation, not overwrite); two-pass collision resolution with
swap/pass-through detection and revert cascades; `known_map` uses −1 =
unknown (Table II was previously untrue — unknown looked free); episode
truncation at `max_steps` with correct SB3 bootstrapping; fully seeded
map-generation/spawning (reproducible); rendering opt-in; no import-time
side effects.

## Files
- `env_paper.py` — environment + seeded MapGen (port of map_gen_v4).
- `test_env_paper.py` — unit tests (`python test_env_paper.py`).
- `train_paper.py` — one training run; paper hyperparameters by default
  (lr 9e-5, ent 0.05, n_steps 2048, batch 64, 64-64 tanh MLP heads);
  writes config.json provenance, best_model.zip by deterministic eval.
- `run_matrix.py` — the 15-run matrix (N2–5 @20×20 + N5 @25×25, 3 seeds).
- `astar_paper.py` — A* baseline with the three meeting-point heuristics
  (geometric median / largest-known-free-component centroid / bbox centre),
  optimistic unknown-free, replan K=5.
- `eval_paper.py` — 20 held-out maps per config, deterministic PPO vs all
  three A* heuristics, per-episode CSV, mean±std summary, paired Wilcoxon
  vs the strongest heuristic.
- `reward_sensitivity_paper.py` — 4 params × 5 levels × 50k steps.

## Paper text that must change to match this code
1. **Eq. (5)** states the bounding *rectangle* product; code (and Table
   III wording) use the bounding **square**: side = max extent,
   area = side². Correct the equation.
2. **Algorithm 1** nests the goal check under the improvement branch;
   code checks the goal every step (otherwise a fleet can sit at a valid
   goal without terminating). Update lines 23–29.
3. **MDP transition text** says "otherwise no robot moves"; code reverts
   only the colliding robots (standard). One-sentence fix in Sec. III-B1.
4. **Hyperparameters**: lr 9e-5 is now true of the training code; the
   original results were produced with 3e-4 — the revision results replace
   them.
5. Goal reward is +100 (Table III), not the legacy +1000; area-decrease
   bonus is flat +20 (not 20+Δ).

## Order of operations
```
python test_env_paper.py
python run_matrix.py            # ~overnight on a single GPU
python eval_paper.py --tag primary
python reward_sensitivity_paper.py --steps 50000 --seeds 0 1 2
```

## Running on the 5090 desktop
```
git clone https://github.com/ShashikaHDS/Multirobot.git
cd Multirobot
git checkout revision-2026-06
pip install -r paper_rev2/requirements.txt   # torch already present on the 5090 box
cd paper_rev2
python test_env_paper.py                     # must print "All tests passed."
python run_matrix.py --n-envs 16             # 16 workers on the desktop CPU
python eval_paper.py --tag primary
python reward_sensitivity_paper.py --steps 50000 --seeds 0 1 2
```
Throughput reference: ~1000 fps with 8 env workers on a 5070 Ti laptop
(the policy is a small MLP, so the env/CPU side dominates — expect the
matrix in well under 6 h on the desktop). `run_matrix.py` is resumable:
re-launch it after any interruption and completed runs are skipped.
Artifacts (`runs_paper/`, `results/`) are gitignored — copy
`results/*.csv` back by hand (or commit them deliberately) when done.
