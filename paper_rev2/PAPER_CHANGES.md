# PAPER_CHANGES — master checklist for the new-journal submission

Every value below is cross-checked against
`runs_paper/final2m/*/seed*/config.json` (the shipped models) and the
results CSVs. Where this document and `manuscript_edits_energy.md`
overlap, this one governs.

---

## 1. The reward mechanism — how to represent it

**Equation (replaces the old Table III bullet list):**

r_t = c_φ · (Φ(s_{t-1}) − Φ(s_t)) + r_step + r_obs·n_obs,t + r_rob·n_rob,t + r_goal·𝟙[goal_t]

where Φ(s) is the fleet's bounding-square area (side = the larger of
the x/y extents of the robot positions; Φ = side²), n_obs,t / n_rob,t
count robots whose move was blocked this step by an obstacle / another
robot, and goal_t is true when Φ ≤ A_th and the anchored square is
obstacle-free.

**Table III (new values):**

| Term | Symbol | Value |
|---|---|---|
| Contraction shaping coefficient | c_φ | 2.0 |
| Step cost (time/energy) | r_step | −0.05 |
| Obstacle collision (per robot) | r_obs | −5 |
| Robot–robot collision (per robot) | r_rob | −5 |
| Goal | r_goal | +100 |
| Success threshold | A_th | 16 (4×4) |
| Episode cap | T_max | 300 |

**Three sentences that must accompany the equation:**
1. *Potential-based form:* "The contraction term is potential-based
   shaping (Ng et al., 1999) with potential −c_φΦ(s); such shaping
   cannot alter the optimal policy, only accelerate learning." (If the
   γ-corrected form c_φ(Φ_{t-1} − γΦ_t) is claimed, use that exactly —
   the implementation uses the undiscounted difference, which is
   near-identical at γ=0.99 but say what is implemented.)
2. *Selection procedure:* "Values were selected by the one-at-a-time
   sensitivity procedure of Sec. IV-F under the rule *maximise success
   subject to not increasing obstacle contacts*; the collision penalty
   is kept at −5 although −1 scores higher on raw success, because the
   weaker penalty increases infeasible waypoint proposals by 19%."
3. *Energy term:* the step cost is the hotel-load component of the
   platform energy model E_i = P_hotel·T + k_move·d_i (see
   manuscript_edits_energy.md §1 for the full paragraph).

**Do NOT carry over:** the old +20-on-new-best / −0.5-on-growth pair.
(They still exist as inert fields in config.json — `area_decrease` /
`area_increase` are ignored whenever `potential_coef ≠ 0`.)

## 2. Old → new value map (what to change, where)

| Item | Old paper draft | New (shipped) | Paper location |
|---|---|---|---|
| Goal reward | +1000 (code) / +100 (draft) | **+100** | Table III |
| Area shaping | +20 on new best / −0.5 growth | **c_φ=2.0 potential shaping** | Table III + Sec. III |
| Step cost | none | **−0.05** | Table III (new row) |
| Collision penalties | −5 (dead code in old env) | **−5, actually applied** | Table III + note |
| Learning rate | 9e-5 (draft claim) | **3e-4** | hyperparameter table |
| Entropy coef | 0.05 constant | **0.05 → 0 linear anneal** | hyperparameter table |
| Budget | 100k–150k / "20 min" | **2M steps per configuration** | Sec. IV setup |
| Training fleet | N=8 (old v5.py) | **one model per N ∈ {2,3,4,5}** | Sec. IV setup |
| Seeds | 1 | **3 per configuration** | Sec. IV setup |
| Unknown cells | =0 (indistinguishable from free) | **= −1** (Table II now true) | Table II |
| Episode termination | success only (unbounded) | **truncation at 300** | Sec. III |
| Evaluation | undisclosed stochastic | **disclosed: 5 seeded stochastic rollouts × 20 held-out maps; deterministic reported as ablation with the livelock analysis** | Sec. IV protocol |
| Other PPO | — | n_steps 2048, batch 64, γ .99, GAE .95, clip .2, vf .5, 64-64 tanh, 16 envs | hyperparameter table |
| Compute | "RTX 5090, Windows 11" | RTX 5090 (Ubuntu, torch 2.11+cu128) for training; ~5,000 env-steps/s; 2M-step run ≈ 14 min | Sec. IV |

## 3. Equation / algorithm / text corrections

1. **Eq. (5):** currently the bounding-rectangle product; the
   implementation (and Table III wording) uses the bounding **square**:
   side = max(Δx, Δy) + 1, A = side². Correct the equation.
2. **Algorithm 1:** the goal check must run every step, not only inside
   the improvement branch (otherwise a fleet already inside a valid
   square never terminates). Update lines ~23–29.
3. **MDP transition sentence:** "otherwise no robot moves" → per-robot
   resolution: only conflicting robots are reverted (same-target,
   swap, and occupied-cell conflicts), non-conflicting robots proceed.
4. **Success criterion sentence:** add "and the anchored square must be
   obstacle-free and within bounds".

## 4. New content to add (each has a ready artifact)

| Addition | Source artifact |
|---|---|
| Evaluation protocol + livelock rationale (why stochastic execution) | results_final_det/ + the 19/19 cycle analysis |
| Classical baseline: A* pipeline, four published meeting-point heuristics incl. Song et al. minimax; McNemar success parity; distance concession | results_final2b (regenerate on final2m if desired), astar_paper.py |
| Frontier-method comparison (colleague's algorithm) — **needs citation from user** | results_frontier/, results_frontier_rl/ (regenerating on final2m now) |
| Difficulty-stratified analysis + mechanism (valid-goal-region scarcity) | results_difficulty/, figures/difficulty_stratified |
| Energy analysis subsection (parametric ρ; balance across power profiles) | results_energy/, figures/energy, manuscript_edits_energy.md §4b |
| Fairness results (Jain, per-robot stacked figure) | results_final2b + figures/per_robot_dist |
| Sensitivity v2 (re-centred at the shipped values) | pending 5090 run → results/reward_sensitivity_v2.csv |
| Reproducibility statement (independent same-seed retrain matches exactly) | final2 vs final3 |
| Training curves | figures/training_curves (final2m, uniform 2M axis) |

## 5. Claims to delete or soften

- "Minimises energy by reducing total distance travelled" → replace per
  manuscript_edits_energy.md §1 (time term + emergent balance).
- Any claim of matching/beating A* on **distance** — the pipeline is
  significantly shorter (Wilcoxon p<0.001); concede and reframe.
- "Collision penalties robust to 2×/0.5×" (old sensitivity) → replaced
  by the v2 sweep + the success/contact trade-off finding.
- Undisclosed stochastic evaluation — now disclosed with rationale.
- "8 robots" anywhere (old training config) → N ∈ {2,…,5}.

## 6. Figure/table inventory (repo file → paper element)

| Paper element | File |
|---|---|
| Fig. training curves | figures/training_curves.pdf |
| Fig. sensitivity (2 rows × 4 params) | figures/sensitivity.pdf (regenerate from v2 CSV when it lands) |
| Fig. energy (worst-share vs ρ; energy-Jain) | figures/energy.pdf |
| Fig. map environments (8 maps) | figures/frontier_rl_maps.pdf |
| Fig. frontier-vs-RL (3 panels) | figures/frontier_rl.pdf |
| Fig. per-robot distances (stacked) | figures/per_robot_dist.pdf |
| Fig. difficulty-stratified (2 panels) | figures/difficulty_stratified.pdf |
| Table: headline results | results_final2m/summary.csv |
| Table: PPO vs A* stats | results_final2b/{wilcoxon,success_tests}.csv |
| Aggregate comparison table | results_frontier_rl/summary.csv |
| Per-map comparison table | results_frontier_rl/per_map.md |
| Per-robot distance table | results_frontier_rl/per_robot.md |
| Table: sensitivity values | results/reward_sensitivity_v2.csv (pending) |

## Open items

1. **Colleague-paper citation** (frontier method) — required, from user.
2. **Sensitivity v2 run** on the 5090 (launch block in README/report).
3. **Power measurement** (optional but recommended): converts energy
   figures to Joules via `energy_analysis.py --p-hotel --p-drive --step-s`.
4. Optionally regenerate results_final2b (A* comparison) on final2m for
   single-model-set purity — expected within noise of the final2 rows.
