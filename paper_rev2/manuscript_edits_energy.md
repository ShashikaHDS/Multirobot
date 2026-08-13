# Manuscript edits: energy treatment (what to change and the ready text)

The current draft claims energy awareness via "reducing the total distance
traveled by each robot," but the original reward contains no distance or
time term. The revision recipe *does* contain an energy term and we have
evidence for the balance claim, so the text below replaces assertion with
mechanism + measurement. Section numbers refer to the current draft.

---

## 1. Section III (reward design) — replace the energy sentence(s)

**Delete** (or equivalent phrasing wherever it appears):
> "A secondary objective is to minimize energy consumption ... achieved by
> reducing the total distance traveled by each robot."

**Insert** (after the reward-term definitions):

> Energy consumption is addressed through the structure of the reward
> rather than through an explicit distance penalty. For the target
> platform, a robot's mission energy is well modelled as
> $E_i = P_{hotel} \, T + k_{move} \, d_i$, where $P_{hotel}$ is the
> constant power drawn by computation and sensing, $T$ the mission
> duration, and $d_i$ the distance travelled by robot $i$. The per-step
> cost $r_{step}$ directly minimises the first, dominant component by
> rewarding early termination, while the potential-based contraction term
> suppresses unproductive motion, since movement that does not reduce the
> fleet's bounding area accrues cost without reward. An explicit
> per-robot motion penalty was evaluated in an ablation and produced no
> further improvement (Section IV-F): balanced effort is an emergent
> property of the bounding-box objective itself, whose area is determined
> only by the extremal robots, so no individual robot can complete the
> contraction alone.

*(If the Smorphi power measurement is done in time, add: "On the Smorphi
platform we measure $P_{hotel} = \dots$ W against $\dots$ W while
driving, so mission time accounts for $\dots\%$ of mission energy at
typical episode lengths." — this single sentence converts the argument
from a model to a measurement.)*

## 2. Table III — new reward values (final recipe)

| Term | Symbol | Value |
|---|---|---|
| Potential-based contraction | $c_\phi\,(A_{t-1}-A_t)$, $c_\phi$ | 2.0 |
| Step cost (time/energy) | $r_{step}$ | −0.05 |
| Obstacle collision (per robot) | $r_{obs}$ | −5 |
| Robot–robot collision (per robot) | $r_{rob}$ | −5 |
| Goal | $r_{goal}$ | +100 |

Plus one sentence: "Values were selected by the sensitivity procedure of
Section IV-F under the rule *maximise success subject to not increasing
obstacle contacts*; the collision penalty is retained at −5 although −1
scores higher on raw success, because the weaker penalty increases
infeasible waypoint proposals by 19%."

## 3. Section IV (results) — the two energy paragraphs

**Fairness (new):**
> Across all configurations the learned policy distributes travel almost
> uniformly (Jain index $J = 0.96$–$0.97$), whereas classical
> meeting-point rules trade fairness against path length: the
> distance-optimal geometric-median rule drops to $J=0.66$–$0.79$
> (paired Wilcoxon, $p \le 3.8\times10^{-6}$), and the fairness-oriented
> minimax rule of Song et al. recovers $J=0.88$–$0.93$ only by
> lengthening paths. The learned policy is not on this trade-off
> frontier: it holds uniform fairness at every fleet size without a
> meeting point ever being selected.

**The honest trade (new):**
> The classical pipeline produces significantly shorter paths in every
> configuration (Wilcoxon $p<0.001$); the learned policy's stochastic
> execution trades path length for heuristic-free operation and uniform
> load distribution. Total distance is therefore reported as the cost
> side of the comparison throughout.

## 4. Claims to delete or soften elsewhere

- Any sentence implying the method minimises *total travelled distance*
  (it does not; the baseline wins that axis).
- "Energy-aware" as a bare adjective → tie it to the mechanism each time:
  time-minimisation + emergent balance.
- Abstract/intro: replace "minimising energy" with "balancing energy
  expenditure across the fleet while minimising mission time".

## 4b. NEW Section IV subsection: "Energy analysis" (references figures/energy.pdf)

> **Energy analysis.** We evaluate mission energy under the platform
> model $E_i = P_{hotel}T + k_{move}d_i$ without committing to specific
> hardware constants: writing $\rho$ for the hotel-load share of the
> energy budget, per-robot energy in normalised units is
> $E_i(\rho) = \rho T + (1-\rho) d_i$, and all quantities are reported
> across the full range $\rho \in [0,1]$. Fig. X(a) plots the
> worst-loaded robot's energy relative to the fleet mean: the learned
> policy's worst robot never exceeds 1.21× the fleet mean at any power
> profile — and this bound is invariant across fleet sizes — whereas
> the classical pipeline's strongest heuristic loads its worst robot up
> to 1.9× (N=3) and 1.75× (N=5) the fleet mean in motion-dominated
> regimes, converging to parity only in the trivial limit $\rho \to 1$
> where energy reduces to shared mission time. Fig. X(b) reports the
> Jain fairness index of per-robot energy at $\rho = 0.5$: the learned
> policy holds $J = 0.993$–$0.996$ in every configuration; the
> distance-optimal geometric-median rule is significantly less fair at
> every fleet size above two ($J = 0.95$–$0.98$, paired Wilcoxon
> $p \le 0.002$), while the fairness-oriented minimax rule of Song et
> al. reaches statistical parity at this operating point
> ($J = 0.98$–$0.99$) — but, as Table Y shows, only by surrendering the
> geometric-median rule's distance advantage. The learned policy's
> balance thus requires neither a heuristic choice nor a favourable
> power profile: it is uniform across $\rho$, fleet size, and map size.
> We again note the complementary trade-off: the classical pipeline
> expends less total energy (shorter paths and missions); the learned
> policy's contribution is the uniform distribution of expenditure,
> obtained without any meeting-point selection. *(When the platform
> power measurement becomes available, the same analysis emits Joules
> via the script's --p-hotel/--p-drive hook — one sentence to add.)*

## 5. Evidence backing each claim (all in the repo)

| Claim | Artifact |
|---|---|
| Step cost = time/energy term, robust | `results/reward_sensitivity.csv` (step_cost panel) |
| Motion penalty adds nothing | move-cost pilot (README, "selection" section) |
| Balance emergent, J=0.96–0.97 | `results_final2b/summary.csv` + Wilcoxon |
| Classical fairness–distance frontier | `results_final2b/summary.csv` (median vs minimax vs bbox) |
| Distance concession | `results_final2b/wilcoxon.csv` |
