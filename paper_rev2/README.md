# paper_rev2 — canonical code for the rendezvous paper revision

## SHIPPED RESULTS: tag `final2` (tuned recipe, RTX 5090) — results_final2/

Recipe: potential shaping **c_phi = 2.0** (γ-correct form), step cost
−0.05, collisions −5/−5, ent 0.05→0, lr 3e-4, 64-64 MLP, 3 seeds.
Stochastic protocol (20 held-out maps × 5 seeded rollouts).

| Config | PPO succ | best A* succ | McNemar | PPO steps | PPO dist | Jain |
|---|---|---|---|---|---|---|
| N2 20×20 | 1.00 | 1.00 | p=1.0 | 21 | 25 | 0.961 |
| N3 20×20 | 0.98 | 1.00 | p=1.0 | 62 | 100 | 0.968 |
| N4 20×20 | 0.96 | 1.00 | p=1.0 | 80 | 163 | 0.966 |
| N5 20×20 | 0.88 | 1.00 | p=1.0 | 113 | 295 | 0.971 |
| N5 25×25 | 0.93 | 1.00 | p=1.0 | 101 | 279 | 0.967 |

vs the previous recipe (results_final/): +3pp at N4, +3pp at N5 20×20,
+6pp at N5 25×25, ~12% fewer steps, ~9% less distance. Success remains
statistically indistinguishable from the strongest A* heuristic in every
configuration (exact McNemar; at most 1 discordant map per seed).

**Reproducibility:** tag `final3` (results_final3/) is an independent
end-to-end retrain with the same seeds on the same machine and matches
final2 metric-for-metric — the fully seeded pipeline reproduces exactly.

**Contact counts:** zero-contact success is rare for BOTH methods (A*
averages 2–5 blocked move attempts per episode because optimistic
planning walks into unknown obstacles; PPO 8–71 because stochastic
execution dithers). Report contacts-per-episode as the honest metric;
a "contact" is a blocked infeasible waypoint proposal, not an impact.

## How the reward values were selected (answers "why these numbers?")

The sweep in `results/reward_sensitivity.csv` (4 parameters x 7-8 levels
x N=3,4,5, one-at-a-time, all others at defaults, every model scored with
the *unperturbed* reward metrics) is the selection procedure, not a
post-hoc robustness check. Selection rule: **maximise success subject to
not increasing obstacle contacts**, since contact counts measure
infeasible waypoint proposals that the low-level controller would have to
reject on hardware.

Evidence for the chosen values (N=5, the binding configuration):

| parameter | chosen | why |
|---|---|---|
| `potential_coef` | **2.0** | 0.85 success at 200k steps vs 0.26 for 0.5 -- a ~7x sample-efficiency gain; degrades again at 5.0, so 1-2 is an interior optimum |
| `step_cost` | **-0.05** | flat across -5..0 at N=3/4 (spread 0.02/0.08); -0.05 is the best N=5 level and is justified physically as the hotel-load energy term |
| `collide_obstacle` | **-5** | -1 scores higher on raw success but a full-budget N=5 pilot showed +19% obstacle contacts (86.5 vs 72.6); -5 retained on the contact constraint |
| `collide_robot` | **-5** | same rule; harsher values (<= -20) collapse success because rendezvous requires tight terminal packing |

Full-budget N=5 pilot (20 maps x 5 stochastic rollouts) isolating the
trade-off:

| recipe | success | steps | distance | Jain | obs contacts | robot contacts |
|---|---|---|---|---|---|---|
| c_phi 0.5, collisions -5 | 0.81 | 141 | 333 | 0.961 | 72.6 | 50.9 |
| c_phi 2.0, collisions -1 | 0.88 | 101 | 260 | 0.953 | 86.5 | 44.1 |

The shipped recipe takes the shaping gain and declines the collision
relaxation.

**Metric note.** An "obstacle collision" here is an *attempted* move into
an occupied cell, which the environment reverts -- the robot never enters
the obstacle. On hardware the RL layer emits a grid waypoint that the
Smorphi controller executes, so these events are infeasible waypoint
proposals (inefficiency), not physical impacts. `cf_success`
(collision-free success) is reported as the headline metric because it
cannot be improved by weakening the collision penalty.

## FINAL RECIPE AND RESULTS (2026-08-05, tag `final`, results_final/)

Training (per run; all values recorded in each run's config.json):
reward = potential-based shaping **0.5·(prev_area − area)** every step
(both signs) + **step cost −0.1** + goal +100 + collisions −5 each;
entropy coefficient **annealed 0.05 → 0** linearly over training;
**lr 3e-4**; PPO MultiInputPolicy 64-64 tanh; N2/N3: 1M steps,
N4/N5: 1.5M, N5@25×25: 2M; 3 seeds each.

Evaluation: **seeded stochastic rollouts** (5/map, 20 held-out maps,
censored means). Rationale: the env is deterministic, so a memoryless
deterministic policy that revisits a joint state livelocks (verified:
19/19 deterministic failures were exact position cycles); sampling is
the symmetry-breaking mechanism. Deterministic numbers are reported as
an ablation (results_final_det/).

| Config | PPO success | best A* success | McNemar p | PPO dist | A* dist |
|---|---|---|---|---|---|
| N2 20×20 | 1.00 | 1.00 | 1.0 | 29.0 | 12.9 |
| N3 20×20 | 0.99 | 1.00 | 1.0 | 93.2 | 19.1 |
| N4 20×20 | 0.93 | 1.00 | 1.0 | 184.7 | 47.2 |
| N5 20×20 | 0.85 | 1.00 | ≥0.25 | 320.9 | 43.4 |
| N5 25×25 | 0.87 | 1.00 | ≥0.5 | 312.2 | 54.3 |

Success is statistically indistinguishable from the steel-manned A*
baseline in every configuration (exact McNemar, all p ≥ 0.25). A* is
significantly shorter on total distance everywhere (Wilcoxon p < 0.001)
and on steps except N4 (p = 0.105) — the honest claim is
success-competitiveness *without any meeting-point heuristic*, not
distance parity.

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
