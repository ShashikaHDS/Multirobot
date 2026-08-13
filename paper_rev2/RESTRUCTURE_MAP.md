# RESTRUCTURE_MAP — root_revised.tex → new-journal submission

Source: `d:\RL\Teal_rl\paper_overleaf\root_revised.tex` (463 lines,
IEEEtran journal class). Disposition verdicts: **KEEP** (verbatim or
light edit) / **REWRITE** (same topic, new content) / **MOVE** /
**DELETE**. Since this is a fresh submission, ALL revision markup goes:
green blocks deleted outright, red wrappers unwrapped to plain text
(lines 1–12 convention comment deleted).

New-paper skeleton (approved plan): 1 Introduction · 2 Related Work ·
3 Problem Formulation · 4 Method · 5 Experimental Setup · 6 Results ·
7 Hardware Experiments · 8 Discussion & Limitations · 9 Conclusion.

---

## Preamble & front matter

| Current (lines) | Verdict | Target / notes |
|---|---|---|
| Revision-convention comment (1–12) | DELETE | fresh submission |
| documentclass/packages (14–22) | KEEP | venue may swap class later |
| Title (26–28) | KEEP (light) | consider adding "energy-balanced": *"…Energy-Balanced Autonomous Rendezvous…"* — decide at the end |
| markboth placeholder (30–31) | REWRITE | proper running head |
| Abstract green (39) | DELETE | |
| Abstract red (40) | REWRITE | update: uniform 2M budget, TWO baselines (A* pipeline + published frontier method), difficulty-stratified analysis, energy-balance across power profiles, livelock finding, stochastic protocol |
| Impact statement (42–44) | REWRITE | fix "minimises energy" → time-minimising + balance framing; add the livelock insight as a community-level takeaway |
| Keywords (48–50) | KEEP | maybe add "energy-aware coordination" |

## I. INTRODUCTION → new §1 + new §2 (Related Work)

| Current | Verdict | Target / notes |
|---|---|---|
| RMR background + coupling mechanisms (54–56) | KEEP, **MOVE to §1 opening (¶1) + §2a** | trim coupling detail by ~30% — it's platform context, not the contribution |
| Rayguru control-layer note (58) | KEEP → §2 | good complementary-work paragraph |
| Coordination literature (60–63) | KEEP → §2a | classical coordination |
| Burgard + energy para (65–70) | REWRITE → §2a | keep Burgard + Pasumarthi; the "no existing approach addresses energy-efficient rendezvous in unknown envs" sentence must be updated — we now cite PIER (Song et al. RA-L 2024), FBR (Tellaroli et al. IROS 2024), and the colleague's method [CITATION PENDING]; the gap becomes "no *learning-based, heuristic-free* rendezvous under partial observability" |
| RL literature (72–73) | KEEP → §2b | |
| Centralised-vs-MARL + discrete-abstraction paras (75) | KEEP → §4 (Method) design-choices | better home than intro |
| Reward-shaping design para (77) | REWRITE → §4 | update to the actual potential-based form (c_φ·ΔΦ), keep the "learns WHERE to converge" framing |
| Contributions paragraph (79) | REWRITE → §1 | replace with the 5-contribution list from the approved plan |
| **NEW** | ADD → §2c | assumptions table: PIER / FBR / colleague / A*-pipeline / ours (map knowledge, comms, meeting-point specified?, learning?) |

## II. ROBOT PLATFORM → new §7 (Hardware) intro + kept early where needed

| Current | Verdict | Target |
|---|---|---|
| II-A Smorphi (84–95) | KEEP → §7.1 | platform description; Fig. smorphi stays |
| RMR-qualification para (88) | KEEP → §7.1 | |
| II-B Docking mechanism (97–101) | KEEP → §7.2 | the "RL only delivers to the cell" para (101) also cited in §4 |
| II-C Autonomy architecture (103–108) | KEEP → §7.3 | |
| Centralised-computation-vs-observability para (108) | KEEP, **MOVE → §3** (Problem Formulation) | it defines the observability model — belongs with the formulation |

Rationale for the move: leading with 1.5 pages of hardware before the
problem statement buries the contribution; §3–6 are self-contained on
the abstract grid problem, §7 grounds it. One forward-reference sentence
in §1 ("targeting the Smorphi platform of §7").

## III. METHOD → new §3 (Formulation) + §4 (Method)

| Current | Verdict | Target / notes |
|---|---|---|
| Section intro + roadmap (112–114) | REWRITE | split across §3/§4 intros |
| III-A objectives/constraints (118) | REWRITE → §3 | **fix the false energy sentence** ("minimize energy…by reducing total distance") → energy model E_i = P_hotel·T + k·d_i + time-term + emergent balance (PAPER_CHANGES §1) |
| FOV r=1 + comms assumptions (120) | KEEP → §3 | |
| Gym/SB3 + PPO-choice para (125) | KEEP → §4 (trim) | 3 reasons for PPO stay |
| MDP formulation (127–136) | KEEP → §3, **fix 𝒫** | transition: per-robot conflict reversion (not "no robot moves") |
| Env + Eq. map (146–150) | KEEP → §3 | add unknown = −1 encoding sentence |
| Action/obs spaces + Tables I–II (152–166, 250–286) | KEEP → §3 | Table II known_map: {0,1,unknown(−1)} |
| **Reward subsection (168–173) + Eq. minarea (169–172)** | **REWRITE → §4** | the core change: Eq. → bounding SQUARE (side = max extent; A = side²); reward → r_t = c_φ(Φ_{t−1}−Φ_t) + r_step + collision terms + goal; Table III new values (c_φ=2.0, −0.05, −5, −5, +100); the three mandatory sentences (shaping invariance, selection rule, energy tie-in) |
| Policy optimisation + Eq. clip (175–180) | KEEP → §4 | |
| **Algorithm 1 (182–236)** | REWRITE → §4 | goal check every step (not nested under improvement); reward lines match new mechanism; add truncation at 300 |
| Walkthrough (238) | REWRITE → §4 | match new algorithm |
| Fig. exploring_map (242–247) | KEEP → §3 | |
| III-C Training: green RTX2060 (319) | DELETE | |
| Training hardware/arch para (320) | REWRITE → §4.4 + §7 | **Windows 11 → actual machines** (RTX 5090/Ubuntu for final runs); 64-64 MultiInputPolicy stays; throughput ≈5,000 steps/s, 2M run ≈ 14 min |
| 2→5 robots + seeds para (322) | REWRITE → §4.4 | uniform **2M steps**, 3 seeds, entropy **schedule** |
| ent_coef subsection (324–325) | REWRITE → §4.4 | 0.05 constant → **0.05→0 linear anneal** + one-line livelock forward-reference |
| **lr subsection (327–328)** | REWRITE → §4.4 | **9e-5 → 3e-4** |
| Other PPO hyperparams (330) | KEEP → §4.4 | values unchanged |
| Training-curves text (332–334) | REWRITE → §6.1 | new Fig (training_curves.pdf, uniform 2M axis); reward-scale caveat |

## IV. RESULTS → new §5 (Setup) + §6 (Results)

| Current | Verdict | Target / notes |
|---|---|---|
| Results intro + 20-rollout protocol para (337–339) | REWRITE → §5 | becomes the full protocol subsection: 20 held-out maps × **5 seeded stochastic rollouts** × 3 seeds; censored means; **livelock justification** (19/19 cycles; det ablation forward-ref); metrics incl. Jain, contacts, revealed% |
| TC01 robots 2–5 (341–349) | REWRITE → §6.2 | now real: per-N models, final2m numbers; old Fig test1_paths can stay as qualitative rollout illustration if regenerated/kept honestly |
| TC02 map size (358–366) | REWRITE → merge into §6.2 | 25×25 row |
| TC03 shapes / TC04 density (389–396) | REWRITE → **§6.4 difficulty-stratified** | replaces both with the systematic version (4 levels × 20 maps + mechanism panel); old figures optionally kept as qualitative examples |
| TC05 initial positions (398–399) | DELETE (fold) | covered by 20 random held-out maps + seeded spawns; one sentence in §5 |
| Old bar-graph figure (351–356) + per-TC bar text | REWRITE | replaced by per-robot stacked figure + per-map tables |
| **TC06 sensitivity (401–402)** | REWRITE → §6.6 | v2 sweep: 4 params × 7–8 levels (wide ranges) × N3/4/5; **delete the "collision penalties are robust" claim — our data refutes it** (success/contact trade-off + boundary collapse); two-row figure |
| Real-world experiments (404–414) | REWRITE → §7.4 | re-shot demo [PLACEHOLDERS until footage]; keep Kinovea methodology; add power measurement + Joules table hook |
| **A* baseline subsection (416–425)** | REWRITE → §5 (setup) + §6.3 | 3→**4 heuristics** (add Song et al. minimax, cite RA-L 2024); goal-validation + stall recovery described; **flip the false claim** "matches or exceeds on total distance" → success parity (McNemar p=1.0) + fairness win + distance concession (Wilcoxon p<0.001) |
| **NEW** | ADD → §6.3 | frontier-method comparison (colleague [CITATION PENDING]): aggregate + per-map + per-robot tables, bimodality finding |
| **NEW** | ADD → §6.5 | energy & fairness subsection (manuscript_edits_energy.md §4b verbatim + energy figure) |
| **NEW** | ADD → §6.7 | deterministic ablation + livelock analysis (results_final_det) |
| Computational requirements (427–429) | REWRITE → §7.5 or §5 | new numbers: 5090/Ubuntu, ~5,000 steps/s, 2M ≈ 14 min, full matrix ≈ 2 h; laptop numbers optional |
| Discussion green (432) | DELETE | |
| Discussion red + limitations list (433–445) | KEEP → §8 (edit) | update: comparison-constrained para now obsolete (we HAVE two baselines) → replace with the when-classical-vs-learned trade-off table; limitations list keeps: comms, localisation, dynamic obstacles, scaling, sample efficiency, hardware tolerances; **add**: OOD density degradation (0.52 on dense maps), contact counts, centralisation scope; reward-design bullet updated (sweep = selection procedure) |

## V. CONCLUSIONS → new §9

| Current | Verdict | Notes |
|---|---|---|
| Green block (448) | DELETE | |
| 4-claim list (449–455) | REWRITE | becomes 5 claims (add energy-balance + livelock); claim 3 reworded to parity-without-heuristics + the difficulty-stratified superiority |
| Future work (457) | KEEP (light edit) | |

## Figures: replace / keep / add

| Current figure | Verdict |
|---|---|
| rl_rmewrk (framework) | KEEP (check reward terms drawn match new mechanism — may need small edit) |
| explore map_new | KEEP |
| training graph all.png | REPLACE → training_curves.pdf |
| test1_paths, test 2/3/4/5 path figs | KEEP as qualitative illustrations (or regenerate rollouts from final2m — decide in P2) |
| Bargraph_5 (per-TC distances) | REPLACE → per_robot_dist.pdf + tables |
| real_exp | REPLACE after re-shoot [PLACEHOLDER] |
| smorphi details | KEEP |
| **ADD** | sensitivity.pdf (v2), energy.pdf, frontier_rl_maps.pdf, frontier_rl.pdf, difficulty_stratified.pdf |

## Cross-cutting rules for the rewrite

1. Strip ALL colour markup; delete every green block.
2. Every number from the artifact CSVs (P3 checker will verify).
3. [CITATION PENDING] markers: colleague's paper; add bib entries for
   Song et al. 2024, Tellaroli et al. 2024, and Jain's fairness index.
4. Keep the paper's existing voice/notation (G, A_min→Φ, P_i) where
   compatible; define Φ once in §3.
5. Supplementary material: per-map tables, deterministic ablation full
   table, sensitivity CSV → mention "supplementary" consistently.
