"""
Generates Revision_Plan.pdf — a structured response to reviewer comments and
prioritized improvement plan for the RL-based multi-robot rendezvous paper.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, ListFlowable, ListItem
)
from datetime import date


OUT = "Revision_Plan.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10, textColor=colors.HexColor("#1a3a6c"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#1a3a6c"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceAfter=4, spaceBefore=8, textColor=colors.HexColor("#333333"))
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=18, bulletIndent=6, spaceAfter=3)
NOTE = ParagraphStyle("NOTE", parent=BODY, backColor=colors.HexColor("#fff5e6"), borderColor=colors.HexColor("#e6a23c"),
                      borderWidth=0.5, borderPadding=6, leftIndent=4, rightIndent=4)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=9, textColor=colors.HexColor("#555555"))
CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier", fontSize=9, leading=11,
                      backColor=colors.HexColor("#f4f4f4"), borderPadding=4)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BULLET):
    return ListFlowable(
        [ListItem(Paragraph(t, style), leftIndent=14, value="bullet") for t in items],
        bulletType="bullet", start="-", leftIndent=12,
    )


def hr():
    t = Table([[""]], colWidths=[16 * cm], rowHeights=[0.5])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888"))]))
    return t


story = []

# ---------------- COVER ----------------
story += [
    P("Revision Plan and Reviewer Response", H1),
    P("Reinforcement Learning-based Autonomous Rendezvous for Docking of "
      "Reconfigurable Modular Robots in Unknown Environments", BODY),
    P(f"Prepared: {date.today().isoformat()}", SMALL),
    Spacer(1, 0.4 * cm),
    P("<b>Purpose.</b> This document maps every reviewer comment to a concrete action, "
      "then consolidates the actions into a prioritized revision plan. It is meant as a "
      "working checklist for the resubmission, not as the rebuttal letter itself — the "
      "rebuttal letter should be derived from Section 1 by paraphrasing the response "
      "into a polite, third-person form.", BODY),
    Spacer(1, 0.2 * cm),
    P("<b>Bottom line.</b> Reviewer 1 and Reviewer 4 recommend acceptance after revisions; "
      "their comments are clarifications and editorial fixes. Reviewer 2 is the threat "
      "to acceptance and must be answered with (a) a stronger novelty framing and "
      "(b) a quantitative comparison against a non-RL planning baseline. Reviewer 3 "
      "is a long generic checklist; most items can be absorbed into a single "
      "Discussion / Limitations subsection, with one substantive addition (reward "
      "sensitivity), which is already prototyped in <font face='Courier'>reward_sensitivity.py</font>.", BODY),
    Spacer(1, 0.3 * cm),
    P("<b>Reproducibility flag.</b> The paper reports learning rate 9&#215;10<sup>-5</sup>, "
      "ent_coef = 0.05, and a goal reward of +100. The closest preserved script "
      "(<font face='Courier'>v7.py</font>) matches the reward table for collision, area decrease, "
      "and area increase but uses goal = +1000 and lr = 3&#215;10<sup>-4</sup>. "
      "No preserved script reproduces all three paper-reported values simultaneously. "
      "Reconciling this is part of the revision (see Section 3).", NOTE),
]

story.append(PageBreak())

# ---------------- SECTION 1: REVIEWER RESPONSES ----------------
story += [P("1. Reviewer-by-reviewer response", H1)]

# ---- Reviewer 1 ----
story += [P("1.1 Reviewer 1 (minor revisions, positive)", H2)]
story += [P("R1 raises five substantive questions plus the suggested citation. Each gets a one-paragraph response in the paper.", BODY)]

story += [P("R1.1 &mdash; Cite Rayguru et al. 2024 on switched adaptive control for self-reconfigurable robots.", H3)]
story += [P("<b>Action:</b> Add to Related Work alongside other RMR control references. One sentence: "
            "\"Rayguru et al. [new ref] introduce switched adaptive control for self-reconfigurable "
            "cleaning robots, addressing adaptability at the control layer; our work is complementary, "
            "addressing adaptability at the coordination layer.\"", BODY)]

story += [P("R1.2 &mdash; How is a Euler-Lagrange system transformed into an MDP?", H3)]
story += [P("<b>Action:</b> Add a paragraph in Section III clarifying the two-layer abstraction. "
            "The Euler-Lagrange dynamics live at the local controller (Smorphi ESP32 + motor controller "
            "in Section II-A). The RL layer operates one level above, where the state is a discrete grid "
            "cell index and an action is a unit-grid waypoint. Translation from waypoint to continuous "
            "torque is performed by the on-board low-level controller. This abstraction is standard in "
            "discrete-action navigation RL (cite e.g., AI2-THOR, MAexp [32]).", BODY)]

story += [P("R1.3 &mdash; Is PPO the right algorithm for a discrete-state discrete-input model?", H3)]
story += [P("<b>Action:</b> Add justification. PPO supports discrete action spaces natively via a "
            "categorical policy head and is the de-facto baseline in the multi-agent navigation literature "
            "(e.g., MAexp [32]). DQN was considered but rejected because (i) the joint action space "
            "grows as 5<sup>N</sup>, making a single Q-head impractical for N>=3, and (ii) on-policy "
            "PPO with a MultiDiscrete head treats the joint as a product of independent categoricals, "
            "which scales linearly in N. Cite the PPO paper (Schulman et al. 2017).", BODY)]

story += [P("R1.4 &mdash; Is PPO the multi-agent variant?", H3)]
story += [P("<b>Action:</b> State plainly that the method is <i>centralized single-agent PPO over the "
            "joint MultiDiscrete action space</i>, not MAPPO or IPPO. Justify via the centralized host "
            "computer described in Section II-C, which already aggregates LiDAR scans into a unified "
            "global map. Acknowledge that decentralized variants are future work.", BODY)]

story += [P("R1.5 &mdash; Where does \"adaptation\" come from?", H3)]
story += [P("<b>Action:</b> Replace \"adaptation\" with the more accurate term \"generalization\" "
            "wherever it appears in the abstract and Section IV. The policy generalizes across map "
            "layouts, obstacle densities, robot counts (within the trained range), and initial "
            "positions because of random domain sampling during training. It does <i>not</i> adapt "
            "online to a single environment in the control-theoretic sense.", BODY)]

story += [P("R1.6 &mdash; Fix Ref [23] (missing details).", H3)]
story += [P("<b>Action:</b> Editorial. Repair the bibliography entry for Boldrer et al. 2023 — add "
            "venue and volume if available.", BODY)]

story += [P("R1.7 &mdash; In what sense is Smorphi an RMR?", H3)]
story += [P("<b>Action:</b> Add one paragraph to Section II-A explaining how the pin-hole docking "
            "from Section II-B makes Smorphi a reconfigurable modular robot in the standard sense: "
            "robots assemble into larger structures and disassemble on command. Reference the prior "
            "docking-mechanism publication. Soften the claim only if the docking paper does not "
            "support \"reconfigurable.\"", BODY)]

story.append(PageBreak())

# ---- Reviewer 2 ----
story += [P("1.2 Reviewer 2 (critical &mdash; main acceptance threat)", H2)]
story += [P("R2's core argument: with a centralized host that aggregates the global map, "
            "classical multi-robot path planning would solve the problem and RL is unnecessary. "
            "This must be answered directly with new experiments and a stronger novelty framing.", BODY)]

story += [P("R2.1 &mdash; Novelty unclear in a centralized setting.", H3)]
story += [P("<b>Action:</b> Reframe the contribution in Section III-A. The RL agent learns "
            "<i>where</i> to converge as a function of the partially-known map; no meeting point is "
            "pre-specified. A classical planner needs a target cell, and choosing one well on a "
            "sparse, partially-explored map is itself a non-trivial decision problem. Our reward "
            "function (area-shrinking) drives convergence without ever specifying a target cell. "
            "This is the contribution.", BODY)]

story += [P("R2.2 &mdash; Why not a classical baseline?", H3)]
story += [P("<b>Action (this is the single largest task in the revision):</b> Add a classical "
            "centralized baseline. Suggested implementation:", BODY)]
story += [bullets([
    "Compute a meeting cell each step from the partially-known map (geometric median of robot "
    "positions, or centroid of the largest connected free component visible so far).",
    "Plan each robot's path with A* on the partially-known map; treat unobserved cells as free "
    "(optimistic) or as unknown-with-cost (conservative); report both.",
    "Re-plan every K steps as the map fills in. Terminate when the bounding area falls below the "
    "same threshold used by the RL agent.",
    "Run on the same Test-Case 1-5 maps with the same seed list; report total distance, max "
    "per-robot distance, success rate, and steps to convergence.",
])]
story += [P("<b>Code reuse:</b> An A* implementation already exists in <font face='Courier'>"
            "test_cuda.py</font> (lines 17-49) but is not wired to the multi-robot environment. "
            "Wrap it in an evaluation loop that consumes the same maps your PPO eval uses.", BODY)]

story += [P("R2.3 &mdash; Energy-efficiency claim is not quantified.", H3)]
story += [P("<b>Action:</b> Replace the qualitative energy claim with a quantitative one. The "
            "v7 environment already tracks <font face='Courier'>distances_traveled</font> per robot. "
            "Aggregate across seeds and report:", BODY)]
story += [bullets([
    "Mean total distance traveled by the fleet (proxy for total energy).",
    "Max per-robot distance (proxy for the worst-loaded robot - this is your balance claim).",
    "Standard deviation across seeds.",
    "Comparison against the baseline from R2.2 on identical maps.",
])]

story += [P("R2.4 &mdash; Polish flagged sentences.", H3)]
story += [P("<b>Action:</b> Rewrite the two sentences R2 quotes verbatim. Specifically: "
            "\"A recent work focused on highlighting...\" and \"This emphasis on fair distribution...\" "
            "Both are run-on / agrammatical. Editorial sweep.", BODY)]

story.append(PageBreak())

# ---- Reviewer 3 ----
story += [P("1.3 Reviewer 3 (16-point checklist &mdash; mostly absorbed into Discussion)", H2)]
story += [P("Most of R3's items can be handled by adding a single Discussion / Limitations "
            "subsection. Items 7 and 14 are off-topic for an RL paper (database normalization, "
            "weight-database sensitivity analysis) and should be addressed by reinterpreting them "
            "as reward sensitivity, which we can answer concretely.", BODY)]

r3_items = [
    ("#1 Full observability assumption.",
     "Clarify partial observability is at the sensor level: each robot only sees within its LiDAR "
     "perimeter; the aggregated map is the union of those scans, not a god-mode view."),
    ("#2 PPO sample inefficiency.",
     "Report total environment steps used per model (e.g., 100k steps per configuration in "
     "current training). Acknowledge sample efficiency as a known PPO trade-off."),
    ("#3 Sim-to-real gap.",
     "Point to the existing Section IV-F real-world experiments. Add one sentence on why the "
     "discrete-grid abstraction transfers: the on-board controller absorbs continuous dynamics."),
    ("#4 Nonlinear scalability in agents.",
     "Frame the joint MultiDiscrete factorization as the main mitigation: the policy head grows "
     "linearly, not exponentially, in N. Discuss that we tested up to N = 5."),
    ("#5 Communication constraints / delays.",
     "Future work. State the assumption explicitly: synchronous, reliable communication via the "
     "centralized host. Add to Limitations."),
    ("#6 Localization assumptions.",
     "Future work. State that SLAM is provided externally (paper Section II-C describes the "
     "EKF-fused IMU+odometry+LiDAR pipeline). RL is conditional on this."),
    ("#7 Dynamic data normalization (off-topic).",
     "Reinterpret as: the observation is normalized to a fixed-shape grid; describe normalization "
     "explicitly in Section III. This is the most charitable read."),
    ("#8 Reward sensitivity / task specificity.",
     "Addressed by new Section IV-X using <font face='Courier'>reward_sensitivity.py</font>. See R3.14."),
    ("#9 Collision risks during exploration.",
     "Already handled via collision penalty (-5). Add one sentence noting that collisions can occur "
     "during exploration but are penalized into rare events at convergence."),
    ("#10 Policy instability in dynamic environments.",
     "Future work. Reference <font face='Courier'>Multirobot_dynamic.py</font> as ongoing work."),
    ("#11 Computational requirements.",
     "Report training wall-clock time and inference time per step (likely sub-millisecond on CPU). "
     "Hardware is already in Section III-C."),
    ("#12 Real-world details / comparisons.",
     "Expand Section IV-F: add the host-computer hardware, the ROS 2 node graph, message rates, "
     "and total wall-clock time per real-world run. The baseline comparison from R2.2 covers the "
     "missing-comparison half."),
    ("#13 Hardware heterogeneity / docking tolerances.",
     "Cite the prior pin-hole docking publication; note that the RL outputs grid-cell targets and "
     "the docking mechanism handles sub-cell pose tolerance."),
    ("#14 Sensitivity analysis / weight database (off-topic, reinterpreted).",
     "Reinterpret as reward sensitivity. Add Section IV-X: a 5x5 sweep over the five reward "
     "parameters (already prototyped in <font face='Courier'>reward_sensitivity.py</font>), "
     "reporting success rate and mean episode length per setting. This is the strongest single "
     "addition we can make for R3."),
    ("#15 Limitations and uncertainty quantification.",
     "Add an explicit Limitations subsection covering the items above. Uncertainty quantification: "
     "report standard deviation across seeds in every results table."),
    ("#16 Conclusion not justified.",
     "Rewrite the conclusion as a numbered list of concrete claims, each backed by a specific "
     "figure or table in Section IV."),
]
for k, v in r3_items:
    story += [P(f"<b>R3 {k}</b> &mdash; {v}", BODY)]

story.append(PageBreak())

# ---- Reviewer 4 ----
story += [P("1.4 Reviewer 4 (positive, mostly editorial)", H2)]

story += [P("R4.1 &mdash; Discuss the reward-shaping connection.", H3)]
story += [P("<b>Action:</b> Add a paragraph to Related Work citing the reward-shaping literature "
            "(Ng et al. 1999, \"Policy invariance under reward transformations\"). Frame the "
            "area-shrinking reward as a domain-specific potential-based shaping that approximates "
            "distance-to-rendezvous without specifying the rendezvous point.", BODY)]

story += [P("R4.2 &mdash; Restructure Section III-B (summary first, then detail).", H3)]
story += [P("<b>Action:</b> Open Section III-B with a one-paragraph summary: \"The method has "
            "three components: (i) a centralized PPO agent over a joint MultiDiscrete action space; "
            "(ii) a partially-observed grid map updated each step by per-robot LiDAR scans; "
            "(iii) an area-shrinking reward that drives convergence without specifying a meeting "
            "point.\" Then expand each in turn. Cross-reference Algorithm 1 steps explicitly "
            "(Step 1, Step 2, ...) in the body.", BODY)]

story += [P("R4.3 &mdash; Provide an equation for the minimum bounding area.", H3)]
story += [P("<b>Action:</b> Add the equation explicitly:", BODY)]
story += [P("A<sub>min</sub>(t) = (max<sub>i</sub> x<sub>i</sub>(t) - min<sub>i</sub> x<sub>i</sub>(t) + 1) "
            "&#215; (max<sub>i</sub> y<sub>i</sub>(t) - min<sub>i</sub> y<sub>i</sub>(t) + 1)", CODE)]
story += [P("where (x<sub>i</sub>(t), y<sub>i</sub>(t)) is the grid position of robot i at step t.", BODY)]

story += [P("R4.4 &mdash; Cite the works that prevent direct comparison.", H3)]
story += [P("<b>Action:</b> In Section IV-G, replace \"most existing methods are designed for "
            "conventional [...]\" with explicit citations. The classical-baseline addition for R2.2 "
            "makes most of this paragraph obsolete &mdash; rewrite it as \"To position our results, "
            "we compare against a classical A*-based rendezvous baseline (Section IV-Y); a direct "
            "comparison with prior RL-based modular-robot rendezvous methods is not possible "
            "because, to our knowledge, no such method exists.\"", BODY)]

story += [P("R4.5 &mdash; Place equations after their text introduction; define variables nearby.", H3)]
story += [P("<b>Action:</b> Editorial sweep. Move every equation to the first paragraph that "
            "discusses it. Define each symbol in the immediately following sentence.", BODY)]

story += [P("R4.6 &mdash; Equations in Fig. 2 must appear in the main text.", H3)]
story += [P("<b>Action:</b> Add the PPO clipped surrogate objective L<sup>CLIP</sup>(theta) to "
            "Section III-B with a brief description, and the reward decomposition R = R<sub>d</sub> "
            "+ R<sub>s</sub> with the meaning of each term.", BODY)]

story += [P("R4.7 &mdash; MDP mentioned in abstract but not in body.", H3)]
story += [P("<b>Action:</b> Add a paragraph in Section III formalizing the MDP as a tuple "
            "&lang;S, A, P, R, gamma&rang;: S = (partially-known map, robot positions); "
            "A = MultiDiscrete([5]<sup>N</sup>) per Table I; P deterministic up to collisions; "
            "R as in Table III; gamma = 0.99 (or whatever value you used).", BODY)]

story += [P("R4 Editorial (items 1-14)", H3)]
story += [bullets([
    "Use possessive voice (\"we\", \"our\") in the main body where appropriate.",
    "Use parenthesis notation (1), (2) ... rather than \"Equation 1\".",
    "Reduce \"(see Fig. X)\" parentheticals &mdash; integrate figures into sentences.",
    "Fix missing \")\" on page 4, column 1.",
    "Define i, j in Eq. 1 (grid indices: i in [0, m), j in [0, n)).",
    "Unify notation: use the same symbol for the action set everywhere (calligraphic A throughout).",
    "Define superscripts in Eq. 3 (a<sup>1</sup> = forward, a<sup>2</sup> = back, a<sup>3</sup> = left, "
    "a<sup>4</sup> = right, a<sup>5</sup> = stay) or list the action names alongside.",
    "Define what a \"position\" is in Eq. 4; link explicitly to the grid G.",
    "Make all figure captions complete sentences and self-contained.",
    "Use \\texttt{} for code-like identifiers such as /map, /goal_pose, /cmd_vel, /slam_out_pose.",
    "Fix malformed straight quotes on page 3 (LaTeX `` and '').",
    "Use booktabs \\midrule between table headers and bodies, or remove inter-row rules.",
    "Add Y-axis labels to all subplots in Fig. 6 (\"Average distance traveled (cells)\").",
    "Cite Stable-Baselines3, OpenAI Gym (or Gymnasium), and the PPO paper (Schulman et al. 2017) "
    "in Section III-B.",
])]

story.append(PageBreak())

# ---------------- SECTION 2: PRIORITIZED IMPROVEMENT PLAN ----------------
story += [P("2. Prioritized improvement plan", H1)]
story += [P("Five tiers, ordered by impact on acceptance. Tiers 1-2 are required to address the "
            "critical reviewers; tiers 3-4 are quality-of-revision improvements; tier 5 is a "
            "reproducibility hygiene item we surface ourselves.", BODY)]

# Tier 1
story += [P("Tier 1 &mdash; Acceptance-critical", H2)]
story += [P("These three items together answer Reviewer 2, which is the only reviewer leaning "
            "toward rejection.", BODY)]
story += [P("T1.1 Add a classical baseline.", H3),
          P("Wire the existing A* in <font face='Courier'>test_cuda.py</font> into a multi-robot "
            "evaluation loop. Use the geometric-median (or centroid-of-largest-free-component) "
            "rendezvous-point heuristic, re-plan every K steps as the map fills, and run on Test "
            "Case 1-5 maps with shared seeds. Output the same metrics for both methods.", BODY)]
story += [P("T1.2 Reframe novelty.", H3),
          P("Rewrite Section III-A around \"the agent learns where to converge as a function of "
            "the partial map\"; emphasize that the RL approach needs no target-cell heuristic.", BODY)]
story += [P("T1.3 Quantify energy.", H3),
          P("Replace qualitative energy claims with three metrics (mean total distance, max "
            "per-robot distance, std across seeds), reported alongside the baseline.", BODY)]

# Tier 2
story += [P("Tier 2 &mdash; Clarifications the reviewers explicitly demanded", H2)]
story += [bullets([
    "T2.1 Formalize the MDP &lang;S, A, P, R, gamma&rang; in Section III (R1.2, R3.1, R4.7).",
    "T2.2 State plainly: single-agent centralized PPO over the joint action space, not MAPPO (R1.4).",
    "T2.3 Add equation for minimum bounding area (R4.3).",
    "T2.4 Discuss reward shaping with citation in Related Work (R4.1).",
    "T2.5 Justify why Smorphi is an RMR (R1.7, R4 implied).",
    "T2.6 Cite Rayguru et al. 2024, fix Ref [23] (R1.1, R1.6).",
])]

# Tier 3
story += [P("Tier 3 &mdash; Reward sensitivity subsection", H2)]
story += [P("Surface the existing <font face='Courier'>reward_sensitivity.py</font> sweep "
            "(5 reward parameters x 5 levels = 25 trained models) as a new Section IV-X. "
            "For each parameter, plot success rate and mean episode length against the swept level, "
            "with the paper default value marked. This single addition answers R3 items 8 and 14, "
            "and complements R4.1's reward-shaping discussion.", BODY)]

# Tier 4
story += [P("Tier 4 &mdash; Limitations / Discussion subsection (absorbs most of R3)", H2)]
story += [P("Add a 1-1.5 page Discussion subsection covering: partial observability framing, "
            "PPO sample efficiency, scaling beyond N=5, communication and localization assumptions, "
            "collision behavior, dynamic-obstacle future work (point to "
            "<font face='Courier'>Multirobot_dynamic.py</font>), training/inference compute, "
            "docking-mechanism tolerance, and uncertainty quantification across seeds.", BODY)]

# Tier 5
story += [P("Tier 5 &mdash; Editorial sweep (R4 items 1-14)", H2)]
story += [P("Single end-to-end pass on the paper PDF after Tiers 1-4 are merged. Estimated half-day.", BODY)]

story.append(PageBreak())

# ---------------- SECTION 3: REPRODUCIBILITY ----------------
story += [P("3. Reproducibility &mdash; reconciling the paper text with the code", H1)]
story += [P("During preparation of this plan we audited the preserved training scripts against the "
            "paper's reported hyperparameters and reward table. No single script matches all paper-"
            "reported values.", BODY)]

repro_data = [
    ["Source", "ent_coef", "learning rate", "collision", "area dec.", "area inc.", "goal"],
    ["Paper (Table III + III-C)", "0.05", "9e-5", "-5", "+20", "-0.5", "+100"],
    ["v5.py / v5_env.py", "0.05", "3e-4", "-2", "+10", "-1", "+10"],
    ["v7.py", "0.05", "3e-4", "-5", "+20", "-0.5", "+1000"],
    ["Final/rl_mltirobot_pygame.py", "(default)", "(default)", "-2", "+1", "-1", "+10"],
    ["reward_sensitivity.py", "0.05", "9e-5", "-5", "+20", "-0.5", "+100"],
    ["train_v8_cnn.py", "0.05", "9e-5", "(paper)", "(paper)", "(paper)", "(paper)"],
]
tbl = Table(repro_data, colWidths=[5.0*cm, 1.6*cm, 1.8*cm, 1.6*cm, 1.7*cm, 1.7*cm, 1.6*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a6c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fa")]),
    ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#fff5e6")),
]))
story += [tbl]

story += [Spacer(1, 0.2*cm)]
story += [P("<b>Interpretation.</b> Only <font face='Courier'>reward_sensitivity.py</font> and "
            "<font face='Courier'>train_v8_cnn.py</font> use the paper-reported hyperparameters. "
            "Both are recent additions and were not the original paper training scripts. The "
            "original training script was likely a now-overwritten edit of <font face='Courier'>v7.py</font>. "
            "We have two options:", BODY)]
story += [bullets([
    "<b>Option A (recommended):</b> Re-train all configurations (N=2,3,4,5; map sizes 20x20 and "
    "25x25) with the paper-stated hyperparameters (lr=9e-5, ent_coef=0.05, goal=+100). "
    "Report fresh numbers. This is the lowest-risk path.",
    "<b>Option B:</b> Update the paper text to match v7.py exactly (goal=+1000, lr=3e-4). Faster "
    "but exposes a different risk: the new numbers might be objected to during peer review of the "
    "revision (\"why did the hyperparameters change?\").",
])]
story += [P("<b>Either way</b>, add a Reproducibility paragraph or supplementary section listing: "
            "exact training script name and git commit, exact hyperparameters, random seeds used "
            "for evaluation maps, and training hardware. This is good practice and pre-empts future "
            "objections.", BODY)]

story.append(PageBreak())

# ---------------- SECTION 4: TIMELINE ----------------
story += [P("4. Suggested order of execution", H1)]
timeline = [
    ["Step", "Task", "Approx. effort"],
    ["1", "Wire A* baseline into eval loop and run on Test Case 1-5 maps (T1.1)", "~5 days"],
    ["2", "Re-train with paper-stated hyperparameters (T1.3, Section 3 Option A)", "~3 days in parallel with step 1"],
    ["3", "Add reward-sensitivity Section IV-X using existing sweep (Tier 3)", "~1 day"],
    ["4", "Rewrite Section III: MDP formalism, novelty, equation, reward shaping (Tier 2)", "~2 days"],
    ["5", "Add Discussion / Limitations subsection (Tier 4)", "~1 day"],
    ["6", "Editorial sweep across whole paper (Tier 5)", "~0.5 day"],
    ["7", "Draft point-by-point rebuttal letter from Section 1 of this plan", "~1 day"],
]
tbl2 = Table(timeline, colWidths=[1.2*cm, 12.5*cm, 3.3*cm])
tbl2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a6c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ALIGN", (0,1), (0,-1), "CENTER"),
    ("ALIGN", (2,1), (2,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fa")]),
]))
story += [tbl2]
story += [Spacer(1, 0.3*cm)]
story += [P("<b>Total:</b> approximately two focused weeks of work. Steps 1 and 2 are the long "
            "pole and can run in parallel. Steps 3-7 are short and sequential.", BODY)]

story += [Spacer(1, 0.4*cm)]
story += [P("End of plan.", SMALL)]

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    title="Revision Plan and Reviewer Response",
    author="Revision planning notes",
)

doc.build(story)
print(f"Wrote {OUT}")
