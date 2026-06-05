"""
Generates Training_Runbook_RTX5090.pdf -- comprehensive operational runbook
for executing the revision training on an RTX 5090.
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

OUT = "Training_Runbook_RTX5090.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10, textColor=colors.HexColor("#1a3a6c"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#1a3a6c"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceAfter=4, spaceBefore=8, textColor=colors.HexColor("#333333"))
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=18, bulletIndent=6, spaceAfter=3)
NOTE = ParagraphStyle("NOTE", parent=BODY, backColor=colors.HexColor("#eaf4ff"), borderColor=colors.HexColor("#1a3a6c"),
                      borderWidth=0.5, borderPadding=6, leftIndent=4, rightIndent=4)
WARN = ParagraphStyle("WARN", parent=BODY, backColor=colors.HexColor("#fff5e6"), borderColor=colors.HexColor("#e6a23c"),
                      borderWidth=0.5, borderPadding=6, leftIndent=4, rightIndent=4)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=9, textColor=colors.HexColor("#555555"))
CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier", fontSize=8.5, leading=11,
                      backColor=colors.HexColor("#f4f4f4"), borderPadding=5, leftIndent=4, rightIndent=4, spaceAfter=6)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BULLET):
    return ListFlowable(
        [ListItem(Paragraph(t, style), leftIndent=14, value="bullet") for t in items],
        bulletType="bullet", start="-", leftIndent=12,
    )


def table(data, col_widths, header_color="#1a3a6c"):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#aaaaaa")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7fa")]),
    ]))
    return t


story = []

# ============================================================
# COVER
# ============================================================
story += [
    P("Training Runbook -- RTX 5090", H1),
    P("Operational guide for executing the revision training run on a single RTX 5090.", BODY),
    P(f"Prepared: {date.today().isoformat()}", SMALL),
    Spacer(1, 0.3*cm),
    P("<b>Scope.</b> This runbook covers the software stack, vectorisation strategy, "
      "training matrix, reproducibility, monitoring, evaluation, and troubleshooting "
      "needed to execute the 40-run revision training matrix on a single RTX 5090 "
      "workstation.", BODY),
    P("<b>Locked choices</b> (from prior planning): paper-stated hyperparameters "
      "(lr = 9&#215;10<sup>-5</sup>, ent_coef = 0.05, goal = +100); 15 primary runs "
      "(N&#8712;{2,3,4,5}, 3 seeds, 100k steps + N=5 at 25&#215;25 with 150k steps) "
      "and 25 reward-sensitivity runs (5 params &#215; 5 levels, 50k steps).", NOTE),
    P("<b>Hardware target.</b> NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32 GB VRAM, "
      "~1700 FP16 TFLOPS). The 5090 is severely underutilised by single-environment "
      "PPO on a 20&#215;20 grid -- the rollout is bottlenecked on Python env stepping, "
      "not GPU compute. Section 4 explains how to exploit this.", WARN),
]
story.append(PageBreak())

# ============================================================
# 1. SOFTWARE STACK
# ============================================================
story += [P("1. RTX 5090 software stack", H1)]
story += [P("The RTX 5090 uses Blackwell architecture with compute capability sm_120. "
            "Older PyTorch wheels (cu118, cu121, cu124) do <b>not</b> include sm_120 kernels, "
            "so training will either fail with \"no kernel image is available\" or silently "
            "fall back to CPU. You must use the cu128 (CUDA 12.8) wheel of PyTorch 2.7 or newer.", BODY)]

story += [P("1.1 Required components", H2)]
stack = [
    ["Component", "Required version", "Notes"],
    ["NVIDIA driver", ">= 565.x", "Driver must support CUDA 12.8 runtime"],
    ["CUDA Toolkit", "12.8 (recommended) or 12.4", "12.4 will work but is slower; 12.8 is native"],
    ["Python", "3.10 - 3.12", "3.13 not yet supported by SB3 stable"],
    ["PyTorch", "2.7.0 or newer, cu128 wheel", "Older wheels lack sm_120 kernels"],
    ["torchvision / torchaudio", "matched to torch 2.7+", "Only needed if you import them"],
    ["Stable-Baselines3", ">= 2.4.0", "Required for gymnasium API and current PPO impl."],
    ["Gymnasium", ">= 0.29", "Replaces deprecated openai gym"],
    ["NumPy", "1.26.x or 2.x", "SB3 2.4+ supports both"],
    ["TensorBoard", ">= 2.15", "For training-curve logging"],
]
story += [table(stack, [3.6*cm, 4.5*cm, 7.2*cm])]

story += [P("1.2 One-command install (Windows PowerShell)", H2)]
story += [P("From inside your project's Python environment:", BODY)]
story += [P("pip install --upgrade pip<br/>"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128<br/>"
            "pip install stable-baselines3 gymnasium tensorboard pandas matplotlib scipy", CODE)]

story += [P("1.3 Stack validation script", H2)]
story += [P("Save the following as <font face='Courier'>scripts/check_stack.py</font> and run it "
            "<i>before</i> launching any training. If any check fails, do not proceed -- you will "
            "burn hours on a stack that silently runs on CPU.", BODY)]

story += [P(
    "import sys, torch, numpy as np<br/>"
    "import stable_baselines3 as sb3, gymnasium as gym<br/>"
    "print(f\"python   : {sys.version.split()[0]}\")<br/>"
    "print(f\"numpy    : {np.__version__}\")<br/>"
    "print(f\"torch    : {torch.__version__}\")<br/>"
    "print(f\"cuda     : {torch.version.cuda}\")<br/>"
    "print(f\"sb3      : {sb3.__version__}\")<br/>"
    "print(f\"gymnasium: {gym.__version__}\")<br/>"
    "assert torch.cuda.is_available(),     \"CUDA not available\"<br/>"
    "assert torch.cuda.device_count() &gt;= 1,  \"No CUDA device\"<br/>"
    "name = torch.cuda.get_device_name(0)<br/>"
    "cap  = torch.cuda.get_device_capability(0)<br/>"
    "print(f\"gpu      : {name}  (sm_{cap[0]}{cap[1]})\")<br/>"
    "assert cap == (12, 0), f\"Expected sm_120 (Blackwell), got sm_{cap[0]}{cap[1]}\"<br/>"
    "x = torch.randn(2048, 2048, device=\"cuda\")<br/>"
    "y = (x @ x).sum().item()<br/>"
    "print(f\"matmul OK on cuda: y={y:.2f}\")<br/>"
    "print(\"STACK OK\")",
    CODE)]

story += [P("Expected output: torch 2.7+, cuda 12.8, sb3 2.4+, gymnasium 0.29+, "
            "gpu \"NVIDIA GeForce RTX 5090 (sm_120)\", and a clean STACK OK line.", BODY)]

story.append(PageBreak())

# ============================================================
# 2. PROJECT LAYOUT
# ============================================================
story += [P("2. Project layout and new files", H1)]
story += [P("All new code lives in new files; legacy paper-cited scripts "
            "(<font face='Courier'>v*.py</font>, <font face='Courier'>rl_*.py</font>, "
            "<font face='Courier'>map_gen_*.py</font>) are not modified.", BODY)]

story += [P("d:/RL/Teal_rl/RL/RL_V2.0/<br/>"
            "&nbsp;&nbsp;env_revision.py            # Clean RectangleReductionEnv with paper Table III<br/>"
            "&nbsp;&nbsp;map_revision.py            # Seedable wrapper around map_gen_v5<br/>"
            "&nbsp;&nbsp;train_revision.py          # PPO training driver (single run)<br/>"
            "&nbsp;&nbsp;astar_baseline.py          # A* + meeting-point heuristic (baseline)<br/>"
            "&nbsp;&nbsp;eval_revision.py           # PPO vs A* rollouts -> results.csv<br/>"
            "&nbsp;&nbsp;plots_revision.py          # CSV -> paper figures<br/>"
            "&nbsp;&nbsp;run_all_revision.ps1       # End-to-end PowerShell driver<br/>"
            "&nbsp;&nbsp;scripts/<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;check_stack.py             # Pre-flight validation<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;monitor_runs.ps1           # nvidia-smi + log tail<br/>"
            "&nbsp;&nbsp;logs_revision/             # Per-run TensorBoard + config.json + model<br/>"
            "&nbsp;&nbsp;results_revision/          # Aggregated CSV + figures<br/>"
            "&nbsp;&nbsp;reward_sensitivity.py      # (already exists -- minor CLI touch-up only)",
            CODE)]

story.append(PageBreak())

# ============================================================
# 3. ENVIRONMENT CONFIG (PPO HYPERPARAMS)
# ============================================================
story += [P("3. Hyperparameters and reward table (frozen)", H1)]

story += [P("3.1 PPO hyperparameters", H2)]
hp = [
    ["Parameter", "Value", "Notes"],
    ["algorithm", "PPO (Stable-Baselines3)", ""],
    ["policy", "MultiInputPolicy", "Dict observation: {known_map, robot_positions}"],
    ["net_arch", "[64, 64]", "Two hidden FC layers, tanh"],
    ["learning_rate", "9e-5", "Paper III-C.2 (locked)"],
    ["ent_coef", "0.05", "Paper III-C.1 (locked)"],
    ["n_steps", "2048", "Rollout buffer per env"],
    ["batch_size", "64", "Minibatch for PPO update"],
    ["n_epochs", "10", "Update epochs per rollout"],
    ["gamma", "0.99", "Discount factor"],
    ["gae_lambda", "0.95", "GAE smoothing"],
    ["clip_range", "0.2", "PPO clip ratio epsilon"],
    ["vf_coef", "0.5", "Value-function loss weight"],
    ["max_grad_norm", "0.5", "Gradient clipping"],
    ["seed", "0, 1, 2", "Three seeds per primary config"],
    ["total_timesteps", "100k or 150k or 50k", "See Section 5 matrix"],
    ["device", "cuda", "Sub-ms inference on 5090"],
]
story += [table(hp, [4.2*cm, 4.3*cm, 6.8*cm])]

story += [P("3.2 Reward table (locked, paper Table III)", H2)]
rew = [
    ["Event", "Reward"],
    ["Bounding area decreases, still above threshold", "+20"],
    ["Bounding area decreases below threshold (goal)", "+100"],
    ["Bounding area increases", "-0.5"],
    ["Collision with wall or obstacle", "-5"],
    ["Collision with another robot", "-5"],
]
story += [table(rew, [12.5*cm, 2.5*cm])]

story.append(PageBreak())

# ============================================================
# 4. VECTORIZATION STRATEGY
# ============================================================
story += [P("4. Vectorisation strategy on the 5090", H1)]
story += [P("This is the most important RTX-5090-specific section. Naive single-environment "
            "PPO on a tiny grid uses about 5-10% of the 5090. Two layers of parallelism are "
            "available and you should use both.", BODY)]

story += [P("4.1 Layer 1: vectorised environments inside a single run", H2)]
story += [P("Use <font face='Courier'>SubprocVecEnv</font> to step many copies of the "
            "environment in parallel within one training process. PPO already supports this "
            "natively: each call to <font face='Courier'>env.step</font> aggregates rollouts "
            "across all worker processes before passing them to the GPU.", BODY)]
story += [bullets([
    "For a $20\\times20$ grid with $N\\in\\{2,3,4,5\\}$, the right number of parallel envs is "
    "<b>16</b>. This saturates the 16 P-cores of a typical 5090 host CPU; beyond 16 you start "
    "spawning extra cores that contend with each other.",
    "Memory footprint per env is &lt; 10 MB, so 16 envs cost &lt; 200 MB host RAM total.",
    "Wall-clock speed-up: about 8&#215; on average vs. a single env, because Python env "
    "stepping is the bottleneck, not GPU compute.",
])]
story += [P("Code skeleton:", BODY)]
story += [P("from stable_baselines3 import PPO<br/>"
            "from stable_baselines3.common.vec_env import SubprocVecEnv<br/>"
            "from env_revision import make_env<br/>"
            "<br/>"
            "env = SubprocVecEnv([make_env(n_robots=4, map_size=20, seed=i) for i in range(16)])<br/>"
            "model = PPO(\"MultiInputPolicy\", env, n_steps=2048, batch_size=64,<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;learning_rate=9e-5, ent_coef=0.05, device=\"cuda\")<br/>"
            "model.learn(total_timesteps=100_000)", CODE)]

story += [P("4.2 Layer 2: multiple training runs in parallel processes", H2)]
story += [P("After Layer 1, a single config still uses only a small fraction of the 32 GB VRAM. "
            "You can run multiple training configurations on the same GPU as separate Python "
            "processes.", BODY)]
story += [bullets([
    "Each PPO process on this network architecture uses about <b>1-1.5 GB of VRAM</b>; the 5090's "
    "32 GB therefore comfortably holds 8-16 concurrent runs.",
    "<b>Recommended:</b> run 4 concurrent processes, each with 16 vectorised envs. This uses "
    "about 6 GB VRAM, 64 worker processes, and keeps the GPU at 60-80% utilisation. Going "
    "to 8 concurrent processes maxes the host CPU before it maxes the GPU.",
    "Always pass a distinct <font face='Courier'>--seed</font> and "
    "<font face='Courier'>--tag</font> to each process so logs and checkpoints do not collide.",
])]
story += [P("Run 4 in parallel via PowerShell (each --& backgrounds the job):", BODY)]
story += [P("Start-Process python -ArgumentList \"train_revision.py --n-robots 4 --map-size 20 --seed 0 --steps 100000 --tag primary\"<br/>"
            "Start-Process python -ArgumentList \"train_revision.py --n-robots 4 --map-size 20 --seed 1 --steps 100000 --tag primary\"<br/>"
            "Start-Process python -ArgumentList \"train_revision.py --n-robots 4 --map-size 20 --seed 2 --steps 100000 --tag primary\"<br/>"
            "Start-Process python -ArgumentList \"train_revision.py --n-robots 5 --map-size 20 --seed 0 --steps 100000 --tag primary\"<br/>"
            "Wait-Process -Name python", CODE)]

story += [P("4.3 What <i>not</i> to do", H2)]
story += [bullets([
    "Do <b>not</b> use DataParallel / DistributedDataParallel -- the policy network is tiny "
    "(under 100 kB of parameters); multi-GPU sharding wastes more on synchronisation than it "
    "saves.",
    "Do <b>not</b> set <font face='Courier'>device=\"cpu\"</font> \"because the env is small\". "
    "The PPO gradient step is still substantial, and CPU PPO is roughly 3&#215; slower on this "
    "workload than cuda PPO with 16 envs.",
    "Do <b>not</b> increase <font face='Courier'>n_steps</font> beyond 4096 -- the rollout buffer "
    "scales linearly in memory, and the gradient variance reduction is marginal past 2048.",
])]

story.append(PageBreak())

# ============================================================
# 5. TRAINING MATRIX WITH 5090 TIMING
# ============================================================
story += [P("5. Training matrix with RTX 5090 timing", H1)]

story += [P("5.1 Primary runs (15 total)", H2)]
prim = [
    ["Config", "N", "Map", "Seeds", "Steps", "Wall (1 env)", "Wall (16 env, 5090)"],
    ["C1", "2", "20x20", "0,1,2", "100k", "~45 min", "~6 min"],
    ["C2", "3", "20x20", "0,1,2", "100k", "~50 min", "~7 min"],
    ["C3", "4", "20x20", "0,1,2", "100k", "~55 min", "~8 min"],
    ["C4", "5", "20x20", "0,1,2", "100k", "~60 min", "~9 min"],
    ["C5", "5", "25x25", "0,1,2", "150k", "~110 min", "~16 min"],
]
story += [table(prim, [1.4*cm, 0.9*cm, 1.6*cm, 1.7*cm, 1.6*cm, 2.5*cm, 3.3*cm])]
story += [P("<b>Total primary wall-clock with 16-env vectorisation, sequential runs:</b> "
            "(6+7+8+9)&#215;3 + 16&#215;3 = 90 + 48 = <b>138 min &asymp; 2.3 h</b>.<br/>"
            "<b>With 4 concurrent processes:</b> &asymp; <b>35-40 min</b>.", BODY)]

story += [P("5.2 Reward-sensitivity sweep (25 runs)", H2)]
story += [P("Each run: N=4, 20&#215;20, single seed, 50k steps. Same 16-env vectorisation. "
            "Per-run wall-clock: ~4 min. Sequential total: ~100 min. With 4 parallel processes: "
            "~25 min. Use the existing <font face='Courier'>reward_sensitivity.py</font> with a "
            "small CLI touch-up to accept --n-envs.", BODY)]

story += [P("5.3 Total time budget", H2)]
budget = [
    ["Phase", "Sequential (16-env)", "4 parallel (16-env each)"],
    ["Primary (15 runs)", "~2.3 h", "~40 min"],
    ["Sensitivity (25 runs)", "~1.7 h", "~25 min"],
    ["Evaluation rollouts (PPO + A*)", "~1 h", "~1 h (CPU-bound)"],
    ["Plot generation", "~5 min", "~5 min"],
    ["<b>Total</b>", "<b>~5 h</b>", "<b>~1.2 h</b>"],
]
story += [table(budget, [4.8*cm, 4.8*cm, 5.4*cm])]
story += [P("Either column fits in a single working session. The 4-parallel column requires a "
            "16-core CPU for full speed-up. If your host has 8 cores, run 2 concurrent processes "
            "and budget accordingly.", BODY)]

story.append(PageBreak())

# ============================================================
# 6. REPRODUCIBILITY
# ============================================================
story += [P("6. Reproducibility checklist", H1)]
story += [P("Every training run writes a <font face='Courier'>config.json</font> next to its "
            "checkpoint so the exact reproduction conditions are preserved.", BODY)]

story += [P("6.1 Seed every randomness source", H2)]
story += [P("import random, numpy as np, torch<br/>"
            "def set_seed(seed: int):<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;random.seed(seed)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;np.random.seed(seed)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;torch.manual_seed(seed)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;torch.cuda.manual_seed_all(seed)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;# env seeding via env.reset(seed=seed) inside make_env<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;# stable-baselines3 also exposes set_random_seed", CODE)]

story += [P("6.2 config.json schema (written by train_revision.py)", H2)]
story += [P("{<br/>"
            "&nbsp;&nbsp;\"git_commit\": \"&lt;sha&gt;\",<br/>"
            "&nbsp;&nbsp;\"torch\": \"2.7.0+cu128\",<br/>"
            "&nbsp;&nbsp;\"sb3\":   \"2.4.0\",<br/>"
            "&nbsp;&nbsp;\"gpu\":   \"NVIDIA GeForce RTX 5090\",<br/>"
            "&nbsp;&nbsp;\"n_robots\": 4, \"map_size\": 20, \"seed\": 0,<br/>"
            "&nbsp;&nbsp;\"total_timesteps\": 100000, \"n_envs\": 16,<br/>"
            "&nbsp;&nbsp;\"learning_rate\": 9e-5, \"ent_coef\": 0.05,<br/>"
            "&nbsp;&nbsp;\"reward\": {\"area_dec\":20,\"area_inc\":-0.5,\"collide\":-5,\"goal\":100},<br/>"
            "&nbsp;&nbsp;\"wall_clock_sec\": 482,<br/>"
            "&nbsp;&nbsp;\"final_eval_success_rate\": 0.94<br/>"
            "}", CODE)]

story += [P("6.3 What to commit to git", H2)]
story += [bullets([
    "All new <font face='Courier'>*_revision.py</font> files.",
    "<font face='Courier'>logs_revision/&lt;run&gt;/config.json</font> for every run "
    "(small text files; commit them).",
    "<font face='Courier'>results_revision/results.csv</font> -- the aggregated evaluation table.",
    "Generated figures in <font face='Courier'>results_revision/figures/</font>.",
    "<b>Do not</b> commit the SB3 model zip files -- they are large; keep them under .gitignore.",
])]

story.append(PageBreak())

# ============================================================
# 7. MONITORING
# ============================================================
story += [P("7. Monitoring during training", H1)]

story += [P("7.1 TensorBoard", H2)]
story += [P("Each run writes to <font face='Courier'>logs_revision/&lt;tag&gt;/&lt;config&gt;/&lt;seed&gt;/tb/</font>. "
            "Launch TensorBoard once at the parent directory:", BODY)]
story += [P("tensorboard --logdir logs_revision --port 6006", CODE)]
story += [P("Curves to watch: <font face='Courier'>rollout/ep_rew_mean</font> "
            "(should rise then plateau), <font face='Courier'>rollout/ep_len_mean</font> "
            "(should fall), <font face='Courier'>train/value_loss</font> (should fall), "
            "and <font face='Courier'>train/entropy_loss</font> "
            "(should stabilise; if it collapses to near-zero too early, policy is over-exploiting).", BODY)]

story += [P("7.2 GPU utilisation", H2)]
story += [P("In a separate PowerShell window:", BODY)]
story += [P("while ($true) { Clear-Host; nvidia-smi; Start-Sleep -Seconds 2 }", CODE)]
story += [P("Healthy signs: GPU-Util &gt; 40% (single run) or &gt; 70% (4 parallel runs); GPU memory "
            "occupied; one Python process per concurrent run. If GPU-Util &lt; 5%, training is "
            "probably stuck on CPU -- re-run the stack check (Section 1.3).", WARN)]

story += [P("7.3 Per-run progress.json", H2)]
story += [P("<font face='Courier'>train_revision.py</font> writes a tiny "
            "<font face='Courier'>progress.json</font> every 5k steps so you can sanity-check "
            "long runs without opening TensorBoard.", BODY)]

story += [P("7.4 Crash recovery", H2)]
story += [P("Save a checkpoint every 25k steps via SB3 "
            "<font face='Courier'>CheckpointCallback</font>. If a process dies, resume from the "
            "latest checkpoint rather than restarting -- this preserves the seed reproducibility "
            "of the final model.", BODY)]

story.append(PageBreak())

# ============================================================
# 8. EVALUATION PIPELINE
# ============================================================
story += [P("8. Evaluation pipeline", H1)]

story += [P("8.1 Inputs", H2)]
story += [bullets([
    "Trained policies in <font face='Courier'>logs_revision/&lt;tag&gt;/.../model.zip</font>.",
    "Test-case maps from <font face='Courier'>Maps/</font> (the 5 maps used in Sections IV-A "
    "through IV-E of the paper) + the same maps regenerated with seeds 0-19 for statistical "
    "aggregation.",
    "A* baseline implementation in <font face='Courier'>astar_baseline.py</font>.",
])]

story += [P("8.2 What eval_revision.py does", H2)]
story += [bullets([
    "Loads each trained policy and rolls it out for 20 episodes per map.",
    "For the same map seeds, runs the A$^{*}$ rendezvous baseline.",
    "Logs per-rollout: total_distance, max_distance, success, steps_to_converge, "
    "inference_time_ms.",
    "Emits a single <font face='Courier'>results.csv</font> with one row per "
    "(method, config, map, seed).",
])]

story += [P("8.3 results.csv schema", H2)]
story += [P("method,config,n_robots,map_size,test_case,map_idx,seed,<br/>"
            "&nbsp;&nbsp;success,steps,total_distance,max_distance,inference_ms", CODE)]

story += [P("8.4 Statistical reporting", H2)]
story += [bullets([
    "Mean &plusmn; std per (method, config) over all seeds.",
    "Paired Wilcoxon signed-rank test for PPO vs A* on each metric per map.",
    "Bootstrap (1000 samples) confidence intervals on the scaling curve (Fig. 6 in the paper).",
    "All statistical tests use <font face='Courier'>scipy.stats</font>.",
])]

story.append(PageBreak())

# ============================================================
# 9. FIGURES AND TABLES
# ============================================================
story += [P("9. Figures and tables to produce", H1)]
fig = [
    ["Output", "Source", "Paper section it replaces"],
    ["Fig. 4 (training curves with std bands)", "TensorBoard logs from primary runs", "Section III-C, Fig. 4 (existing)"],
    ["Fig. 6 (scaling bar chart with error bars)", "results.csv aggregated by config", "Section IV, Fig. 6 (existing)"],
    ["New Fig: PPO vs A* per-map comparison", "results.csv pivoted by method", "Section IV-G (new)"],
    ["New Fig: reward sensitivity heatmap", "Sensitivity sweep aggregated", "Section IV-F (new)"],
    ["Table: configuration summary", "config.json files", "Section III-C (new)"],
    ["Table: compute budget", "wall_clock_sec fields", "Section IV-H (new)"],
    ["Table: paired statistical tests", "results.csv + scipy", "Section IV-G (new)"],
]
story += [table(fig, [5.5*cm, 5.5*cm, 4.0*cm])]

story.append(PageBreak())

# ============================================================
# 10. TROUBLESHOOTING
# ============================================================
story += [P("10. Troubleshooting", H1)]

story += [P("10.1 Symptom: \"no kernel image is available for execution on the device\"", H3)]
story += [P("Cause: PyTorch wheel does not include sm_120 kernels (you have cu118 / cu121 / cu124 "
            "instead of cu128). Fix: reinstall torch from the cu128 index URL "
            "(Section 1.2). Verify with the stack check.", BODY)]

story += [P("10.2 Symptom: training runs but GPU-Util is 0-5%", H3)]
story += [P("Cause 1: device defaulted to CPU. Print "
            "<font face='Courier'>model.device</font> after constructing PPO; it must be "
            "<font face='Courier'>cuda:0</font>. Cause 2: only one environment is being stepped "
            "(no SubprocVecEnv); the env-stepping bottleneck masks GPU activity. Use 16-env "
            "vectorisation as in Section 4.1.", BODY)]

story += [P("10.3 Symptom: \"CUDA out of memory\" with one run", H3)]
story += [P("Unlikely on a 32 GB 5090 with this architecture. If it happens, you probably left "
            "stale tensor handles -- run only one PPO process at a time and confirm "
            "<font face='Courier'>nvidia-smi</font> shows zero VRAM use before launching.", BODY)]

story += [P("10.4 Symptom: episode reward starts high then collapses", H3)]
story += [P("Indicates entropy collapse or value-function explosion. Confirm "
            "<font face='Courier'>ent_coef = 0.05</font> (not zero). Confirm reward magnitudes "
            "match Table III exactly; a stray +1000 from legacy code will dominate the gradient.", BODY)]

story += [P("10.5 Symptom: episode reward never rises", H3)]
story += [P("Confirm the bounding-area reward is being assigned (set a breakpoint in step() and "
            "print). Confirm at least 50% of episodes finish in goal -- if every episode ends in "
            "timeout, the threshold may be unreachable for the current map size; lower it to "
            "16 for 20&#215;20 and 25 for 25&#215;25.", BODY)]

story += [P("10.6 Symptom: A* baseline beats PPO on every map", H3)]
story += [P("Not a bug -- it can happen if the meeting-point heuristic is well-chosen for the "
            "specific maps. Report it honestly. The contribution framing (\"no meeting-point "
            "heuristic required\") still holds.", BODY)]

story += [P("10.7 Symptom: 4 parallel processes are slower than sequential", H3)]
story += [P("CPU contention from too many environment workers. Reduce per-run "
            "<font face='Courier'>n_envs</font> from 16 to 8, or drop concurrency from 4 to 2.", BODY)]

story.append(PageBreak())

# ============================================================
# 11. END-TO-END RUNBOOK
# ============================================================
story += [P("11. End-to-end runbook", H1)]
story += [P("Execute these steps in order. Each step assumes the previous succeeded.", BODY)]

steps = [
    ["#", "Step", "Estimated time"],
    ["1", "Pre-flight: run <font face='Courier'>python scripts/check_stack.py</font>. "
          "Confirm STACK OK, sm_120 detected.", "1 min"],
    ["2", "Write <font face='Courier'>env_revision.py</font> and "
          "<font face='Courier'>map_revision.py</font>. Unit-test with a 100-step rollout.", "~3 h"],
    ["3", "Write <font face='Courier'>train_revision.py</font>. Smoke-test with "
          "<font face='Courier'>--steps 5000 --n-robots 2 --n-envs 4</font>; "
          "confirm config.json and model.zip are written, TensorBoard shows curves.", "~2 h"],
    ["4", "Write <font face='Courier'>astar_baseline.py</font> and "
          "<font face='Courier'>eval_revision.py</font>; smoke-test by rolling out the "
          "smoke-test model and the A* baseline on one map.", "~3 h"],
    ["5", "Touch up <font face='Courier'>reward_sensitivity.py</font> to accept "
          "<font face='Courier'>--n-envs</font>.", "~30 min"],
    ["6", "Launch full training: "
          "<font face='Courier'>powershell -File run_all_revision.ps1</font>. "
          "It runs 4 PPO processes in parallel until the matrix completes.",
          "~1.2 h (4-par) /<br/>~5 h (sequential)"],
    ["7", "Launch <font face='Courier'>python eval_revision.py --tag primary --episodes 20 "
          "--baseline astar</font>. Emits results.csv.", "~1 h"],
    ["8", "Launch <font face='Courier'>python plots_revision.py --results-csv results.csv "
          "--out-dir results_revision/figures/</font>. Emits Fig. 4, Fig. 6, and the two new "
          "figures.", "~5 min"],
    ["9", "Drop the new figures into Overleaf; replace the placeholder text in "
          "<font face='Courier'>root_revised.tex</font> sections IV-F, IV-G, IV-H with the "
          "actual numbers from results.csv.", "~2 h"],
    ["10", "Re-compile root_revised.tex; visual diff vs. the original to confirm the changes "
           "look right.", "~10 min"],
    ["11", "Draft the rebuttal letter from Section 1 of Revision_Plan.pdf in third-person form.",
           "~3 h"],
    ["12", "Submit.", ""],
]
story += [table(steps, [0.9*cm, 12.0*cm, 2.1*cm])]

story += [Spacer(1, 0.3*cm)]
story += [P("<b>Critical-path summary:</b> the entire revision (code, training, evaluation, "
            "figures, paper update, rebuttal) fits in <b>~3 working days</b> on this 5090 "
            "workstation, with most of the wall-clock spent on writing rather than computing.", NOTE)]

story += [Spacer(1, 0.4*cm)]
story += [P("End of runbook.", SMALL)]

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    title="Training Runbook - RTX 5090",
    author="Revision planning notes",
)
doc.build(story)
print(f"Wrote {OUT}")
