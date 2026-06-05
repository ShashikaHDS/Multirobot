#!/usr/bin/env bash
# Launches the full revision training matrix on the RTX 5090 (Linux).
#
# Strategy:
#   - 16 vectorised envs per process (SubprocVecEnv).
#   - Up to 4 PPO processes in parallel (parameterise via CONCURRENCY env var).
#   - Primary: N in {2,3,4,5} x seeds {0,1,2} at 20x20 + N=5 at 25x25 = 15 runs.
#
# Usage:
#   bash run_all_revision.sh             # primary only, concurrency = 4
#   CONCURRENCY=2 bash run_all_revision.sh
#   bash run_all_revision.sh primary     # primary only
#   bash run_all_revision.sh eval        # eval only
#   bash run_all_revision.sh all         # primary + eval + plots

set -e
PHASE="${1:-all}"
CONCURRENCY="${CONCURRENCY:-4}"
TAG="${TAG:-primary}"
LOGDIR="logs_revision"

mkdir -p "$LOGDIR" results_revision

run_one() {
  local n="$1" m="$2" s="$3" steps="$4"
  local label="N${n}_M${m}_seed${s}"
  echo "[start] $label  steps=$steps"
  python train_revision.py \
      --n-robots "$n" --map-size "$m" --seed "$s" --steps "$steps" \
      --n-envs 16 --tag "$TAG" --logdir "$LOGDIR" \
      > "$LOGDIR/${label}.stdout" 2> "$LOGDIR/${label}.stderr"
  echo "[done ] $label"
}

# Throttled parallel launcher.
wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do
    sleep 2
  done
}

run_primary() {
  echo "== Primary matrix (15 runs, concurrency=$CONCURRENCY) =="
  # N in {2,3,4,5} on 20x20, 3 seeds, 100k steps
  for n in 2 3 4 5; do
    for s in 0 1 2; do
      wait_for_slot
      run_one "$n" 20 "$s" 100000 &
    done
  done
  # N=5 on 25x25, 3 seeds, 150k steps
  for s in 0 1 2; do
    wait_for_slot
    run_one 5 25 "$s" 150000 &
  done
  wait
  echo "== Primary matrix complete =="
}

run_eval() {
  echo "== Evaluation (PPO + A*) =="
  python eval_revision.py --tag "$TAG" --logdir "$LOGDIR" \
                          --out results_revision/results.csv
}

run_plots() {
  echo "== Plotting =="
  python plots_revision.py --results-csv results_revision/results.csv \
                           --out-dir results_revision/figures
}

case "$PHASE" in
  primary) run_primary ;;
  eval)    run_eval ;;
  plots)   run_plots ;;
  all)     run_primary; run_eval; run_plots ;;
  *) echo "Unknown phase: $PHASE"; exit 2 ;;
esac
