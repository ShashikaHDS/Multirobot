"""
Generates Training_Plan.pdf — operational plan for the revision training run.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, ListFlowable, ListItem
)
from datetime import date

OUT = "Training_Plan.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10, textColor=colors.HexColor("#1a3a6c"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#1a3a6c"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceAfter=4, spaceBefore=8, textColor=colors.HexColor("#333333"))
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=18, bulletIndent=6, spaceAfter=3)
NOTE = ParagraphStyle("NOTE", parent=BODY, backColor=colors.HexColor("#eaf4ff"), borderColor=colors.HexColor("#1a3a6c"),
                      borderWidth=0.5, borderPadding=6, leftIndent=4, rightIndent=4)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=9, textColor=colors.HexColor("#555555"))
CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier", fontSize=9, leading=12,
                      backColor=colors.HexColor("#f4f4f4"), borderPadding=5, leftIndent=4, rightIndent=4)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BULLET):
    return ListFlowable(
        [ListItem(Paragraph(t, style), leftIndent=14, value="bullet") for t in items],
        bulletType="bullet", start="-", leftIndent=12,
    )


story = []

# ---------------- COVER ----------------
story += [
    P("Training Plan &mdash; Revision Run", H1),
    P("Reinforcement Learning-based Autonomous Rendezvous for Docking of "
      "Reconfigurable Modular Robots in Unknown Environments", BODY),
    P(f"Prepared: {date.today().isoformat()}", SMALL),
    Spacer(1, 0.3 * cm),
    P("<b>Purpose.</b> Operational plan for the new training run that supports the journal "
      "revision. This document fixes hyperparameters, the training matrix, file structure, "
      "and execution commands so that the experiments are reproducible and the resulting "
      "numbers flow directly into the revised paper.", BODY),
    Spacer(1, 0.2 * cm),
    P("<b>Locked choices</b><br/>"
      "&bull; Hyperparameters: paper-stated &mdash; lr = 9&#215;10<sup>-5</sup>, ent_coef = 0.05, "
      "goal reward = +100.<br/>"
      "&bull; Scope: 15 primary runs + 25 reward-sensitivity runs = 40 runs total. "
      "Approximately 9 hours wall-clock on the RTX 5090. No N&gt;5 stretch.<br/>"
      "&bull; All new code lives in new files; legacy scripts "
      "(<font face='Courier'>v*.py</font>, <font face='Courier'>rl_*.py</font>, "
      "<font face='Courier'>map_gen_*.py</font>) are left untouched.", NOTE),
    Spacer(1, 0.3 * cm),
    P("<b>What this run delivers for the revision</b>", H3),
]
story += [bullets([
    "Reproducibility: all reported numbers regenerated from a single command, with a saved "
    "<font face='Courier'>config.json</font> per run. Closes the gap between paper text and code.",
    "Uncertainty quantification: mean &plusmn; std across 3 seeds in every results table "
    "(answers R3 #15).",
    "Quantitative energy claim: total distance and max per-robot distance per config, with "
    "error bars (answers R2 #3).",
    "Reward sensitivity: 5 parameters &#215; 5 levels (answers R3 #8 and #14, and supports R4 #1).",
    "Compute disclosure: wall-clock training and inference times reported per config "
    "(answers R3 #2 and #11).",
])]

story.append(PageBreak())

# ---------------- SECTION 1: TRAINING MATRIX ----------------
story += [P("1. Training matrix", H1)]

story += [P("1.1 Primary runs (15 total)", H2)]
primary = [
    ["Config", "N robots", "Map size", "Seeds", "Steps", "Notes"],
    ["C1", "2", "20×20", "{0,1,2}", "100k", "Smallest fleet"],
    ["C2", "3", "20×20", "{0,1,2}", "100k", "Matches Fig. 4(a)"],
    ["C3", "4", "20×20", "{0,1,2}", "100k", "Matches Fig. 4(b)"],
    ["C4", "5", "20×20", "{0,1,2}", "100k", "Matches Fig. 4(c)"],
    ["C5", "5", "25×25", "{0,1,2}", "150k", "Map-size scalability (Test Case 2)"],
]
tbl = Table(primary, colWidths=[1.5*cm, 2.2*cm, 2.3*cm, 2.5*cm, 1.8*cm, 5.5*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a6c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ALIGN", (0,1), (-2,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fa")]),
]))
story += [tbl]
story += [Spacer(1, 0.2*cm)]
story += [P("<b>Effort:</b> ~20 min per 100k-step run on the 5090; C5 at ~30 min. "
            "Sub-total: <b>~5 hours</b>.", BODY)]

story += [P("1.2 Reward sensitivity sweep (25 runs)", H2)]
story += [P("Five reward parameters &#215; five perturbation levels, single seed each, 50k steps. "
            "Driven by the existing <font face='Courier'>reward_sensitivity.py</font> (paper "
            "hyperparameters already correct in that file).", BODY)]
sens = [
    ["Parameter", "Paper default", "Levels swept"],
    ["area_decrease", "+20", "{+5, +10, +20, +40, +80}"],
    ["area_increase", "-0.5", "{-2.0, -1.0, -0.5, -0.25, -0.1}"],
    ["collide_obstacle", "-5", "{-20, -10, -5, -2, -1}"],
    ["collide_robot", "-5", "{-20, -10, -5, -2, -1}"],
    ["goal", "+100", "{+25, +50, +100, +200, +500}"],
]
tbl = Table(sens, colWidths=[4.5*cm, 3.0*cm, 8.3*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a6c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ALIGN", (1,1), (-1,-1), "LEFT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fa")]),
]))
story += [tbl]
story += [Spacer(1, 0.2*cm)]
story += [P("<b>Effort:</b> 25 runs &#215; ~10 min = <b>~4 hours</b>. Sensitivity sweep runs on "
            "N=4, 20&#215;20 grid (the median primary config).", BODY)]

story += [P("1.3 Total budget", H2)]
story += [P("Primary 5 h + sensitivity 4 h &asymp; <b>9 hours wall-clock</b>. Add another 1-2 hours "
            "for evaluation rollouts (no training) once policies are saved. Whole run fits "
            "comfortably in a single overnight session.", BODY)]

story.append(PageBreak())

# ---------------- SECTION 2: HYPERPARAMETERS ----------------
story += [P("2. Hyperparameters (locked, paper-stated)", H1)]

hp = [
    ["Parameter", "Value", "Source"],
    ["Algorithm", "PPO (Stable-Baselines3)", "Paper III-B"],
    ["Policy", "MultiInputPolicy", "Matches v7.py"],
    ["learning_rate", "9e-5", "Paper III-C.2"],
    ["ent_coef", "0.05", "Paper III-C.1"],
    ["n_steps", "2048", "SB3 default"],
    ["batch_size", "64", "SB3 default"],
    ["n_epochs", "10", "SB3 default"],
    ["gamma", "0.99", "SB3 default; declare in paper MDP tuple"],
    ["gae_lambda", "0.95", "SB3 default"],
    ["clip_range", "0.2", "SB3 default"],
    ["Device", "cuda (RTX 5090, cu128 build)", "Hardware"],
    ["Total steps per run", "100k (primary) / 50k (sensitivity) / 150k (C5)", "See matrix"],
]
tbl = Table(hp, colWidths=[4.5*cm, 5.5*cm, 5.8*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a6c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ALIGN", (1,1), (-1,-1), "LEFT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fa")]),
]))
story += [tbl]

story += [P("2.1 Reward table (locked, paper Table III)", H2)]
rew = [
    ["Event", "Reward"],
    ["Bounding-area decrease (below previous min, above threshold)", "+20"],
    ["Bounding-area decrease (below threshold = goal)", "+100"],
    ["Bounding-area increase", "-0.5"],
    ["Collision with obstacle", "-5"],
    ["Collision with another robot", "-5"],
]
tbl = Table(rew, colWidths=[12*cm, 2.5*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a6c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fa")]),
]))
story += [tbl]

story.append(PageBreak())

# ---------------- SECTION 3: NEW FILES ----------------
story += [P("3. New files to create (no legacy modifications)", H1)]

files = [
    ("env_revision.py",
     "Clean re-implementation of v7's RectangleReductionEnv using the gymnasium API. "
     "Rewards exactly per Table III. Per-robot distance counters exposed via the info dict. "
     "Deterministic seeding via gymnasium's seed() and reset(seed=...). Same observation/action "
     "spaces as v7 so the policy architecture remains comparable to the paper's network. "
     "Imports the connectivity-checked map generator from map_revision.py."),
    ("map_revision.py",
     "Thin wrapper that calls map_gen_v5.py's existing logic but exposes seed and obstacle-density "
     "as arguments. Guarantees BFS connectivity of free cells before returning. Does not modify "
     "map_gen_v5.py."),
    ("train_revision.py",
     "CLI driver. Arguments: --n-robots, --map-size, --seed, --steps, --tag. Writes model and "
     "config.json to logs_revision/<tag>/<config>/<seed>/. Uses paper hyperparameters as defaults. "
     "Records wall-clock training time and total environment steps in config.json."),
    ("eval_revision.py",
     "Loads a saved policy and rolls out 50 episodes on each of the Test Case 1-5 maps with shared "
     "seeds. Logs total distance, max per-robot distance, success flag, steps to convergence, and "
     "mean inference time per step. Emits one row per (method, config, map, seed) into results.csv. "
     "Also runs the A* baseline (Section 5 below) on the identical maps."),
    ("astar_baseline.py",
     "A* planner with three meeting-point heuristics: geometric median of robot positions, centroid "
     "of the largest connected free component of the known map, and current bounding-box center. "
     "Operates on the partially-known map (unknown cells treated as free). Re-plans every K = 5 "
     "steps as the map fills in. Terminates when the bounding area reaches the same threshold as PPO."),
    ("plots_revision.py",
     "Reads results.csv and reward-sensitivity CSV, produces matplotlib figures: scaling curve with "
     "error bars (Fig. 6a revisited), PPO vs A* per-map comparison (new figure), sensitivity heatmap "
     "(new figure). All figures saved as both PDF and PNG for paper inclusion."),
    ("run_all_revision.ps1",
     "PowerShell driver that sequentially launches all 40 training commands and then the eval "
     "rollouts. Logs to revision_run.log. Single command kicks off the whole run."),
]
for fname, desc in files:
    story += [P(f"<b><font face='Courier'>{fname}</font></b>", BODY),
              P(desc, BODY),
              Spacer(1, 0.15*cm)]

story.append(PageBreak())

# ---------------- SECTION 4: CLI RECIPES ----------------
story += [P("4. Reference commands", H1)]

story += [P("4.1 Single primary run", H2)]
story += [P("python train_revision.py --n-robots 4 --map-size 20 --seed 0 --steps 100000 "
            "--tag primary_v1", CODE)]

story += [P("4.2 Whole primary matrix (PowerShell)", H2)]
story += [P("foreach ($n in 2,3,4,5) {\n"
            "  foreach ($s in 0,1,2) {\n"
            "    python train_revision.py --n-robots $n --map-size 20 --seed $s --steps 100000 --tag primary_v1\n"
            "  }\n"
            "}\n"
            "foreach ($s in 0,1,2) {\n"
            "  python train_revision.py --n-robots 5 --map-size 25 --seed $s --steps 150000 --tag primary_v1\n"
            "}", CODE)]

story += [P("4.3 Sensitivity sweep", H2)]
story += [P("python reward_sensitivity.py --n-robots 4 --map-size 20 --steps 50000 --tag sens_v1", CODE)]
story += [P("Note: <font face='Courier'>reward_sensitivity.py</font> already uses the paper "
            "hyperparameters; we may need to add the CLI flags shown above as a small touch-up.", BODY)]

story += [P("4.4 Evaluation rollouts", H2)]
story += [P("python eval_revision.py --tag primary_v1 --episodes 50 --baseline astar", CODE)]

story += [P("4.5 Plots and tables", H2)]
story += [P("python plots_revision.py --results-csv results.csv --out-dir figures_revision/", CODE)]

story += [P("4.6 One-shot end-to-end", H2)]
story += [P("powershell -File run_all_revision.ps1", CODE)]

story.append(PageBreak())

# ---------------- SECTION 5: EVAL + BASELINE ----------------
story += [P("5. Evaluation protocol (apples-to-apples vs A*)", H1)]

story += [P("5.1 Test case maps", H2)]
story += [P("Re-use the five test-case maps from the original paper (Section IV-A through IV-E). "
            "Each map is loaded from a fixed seed so that PPO and A* see identical environments. "
            "For each map: 50 rollouts with episode seeds 0&hellip;49.", BODY)]

story += [P("5.2 Metrics (logged per rollout)", H2)]
story += [bullets([
    "<b>total_distance</b> &mdash; sum of grid cells moved across the fleet (proxy for total energy).",
    "<b>max_distance</b> &mdash; max per-robot distance (proxy for the worst-loaded robot &mdash; "
    "your balance claim).",
    "<b>success</b> &mdash; bounding area reached the threshold before the step cap.",
    "<b>steps_to_converge</b> &mdash; if successful, number of environment steps to threshold.",
    "<b>inference_time_ms</b> &mdash; mean policy / planner step time, on CPU and on GPU.",
])]

story += [P("5.3 Statistical reporting", H2)]
story += [bullets([
    "Aggregate over the three training seeds and 50 episode seeds: mean &plusmn; std.",
    "Paired Wilcoxon signed-rank test per map for PPO vs A* on each metric.",
    "Confidence intervals on the scaling curve (Fig. 6a revisited) via 1000-sample bootstrap.",
])]

story += [P("5.4 A* baseline details", H2)]
story += [bullets([
    "Three meeting-point heuristics evaluated; report the strongest result for A* (steel-manning "
    "the baseline so the comparison is fair).",
    "Re-plan period K = 5 environment steps.",
    "Unknown cells optimistic (treated free) in the planner; report a conservative variant in the "
    "supplementary material if time allows.",
    "Same termination threshold as PPO: bounding area &le; threshold.",
])]

story.append(PageBreak())

# ---------------- SECTION 6: TIMELINE ----------------
story += [P("6. Execution order", H1)]
timeline = [
    ["#", "Step", "Effort"],
    ["1", "Write env_revision.py and map_revision.py; unit-test with a 10-step rollout.", "~3 h"],
    ["2", "Write train_revision.py; smoke-test with --steps 5000 --n-robots 2.", "~2 h"],
    ["3", "Write astar_baseline.py and eval_revision.py; smoke-test on one saved policy.", "~3 h"],
    ["4", "Touch up reward_sensitivity.py for new CLI flags (no logic change).", "~1 h"],
    ["5", "Kick off run_all_revision.ps1; let it run overnight (~9 h).", "overnight"],
    ["6", "Run eval rollouts (PPO and A*) on all saved policies.", "~1.5 h"],
    ["7", "Generate figures and tables via plots_revision.py.", "~1 h"],
    ["8", "Update the paper text (Section III MDP, Section IV-X sensitivity, Section IV-Y "
          "baseline comparison) with new numbers.", "~1 day"],
]
tbl = Table(timeline, colWidths=[1*cm, 12.7*cm, 3.2*cm])
tbl.setStyle(TableStyle([
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
story += [tbl]

story += [Spacer(1, 0.3*cm)]
story += [P("<b>Total:</b> ~3 days of focused coding + an overnight training run + ~1 day of "
            "paper writing. The training itself is unattended.", BODY)]

story += [P("6.1 Risks and mitigations", H2)]
story += [bullets([
    "<b>New PPO numbers differ from Fig. 4 curves in the paper.</b> Plan: regenerate Fig. 4 from "
    "the new TensorBoard logs; this is a few hours of work in plots_revision.py.",
    "<b>A* baseline wins on some maps.</b> Acceptable and honest. Report it; the PPO contribution "
    "is then framed as comparable performance without a target-cell heuristic.",
    "<b>Training is slower than estimated.</b> Drop the C5 25&#215;25 config to 100k steps if "
    "needed; the smaller-map runs are non-negotiable.",
    "<b>Sensitivity sweep takes longer than 4 h.</b> Drop sweep steps from 50k to 30k; relative "
    "ranking is what matters, not absolute convergence.",
])]

story += [Spacer(1, 0.4*cm)]
story += [P("End of plan.", SMALL)]

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    title="Training Plan - Revision Run",
    author="Revision planning notes",
)
doc.build(story)
print(f"Wrote {OUT}")
