#!/usr/bin/env bash
# Locate the Isaac Sim python and run the zero-shot smoke validation.
#
#   bash run_smoke.sh              # smoke: 2 maps, N=4, ~5 min
#   bash run_smoke.sh full         # full protocol, ~1-3 h
#   bash run_smoke.sh noise        # localisation-noise sweep
#
# Detection order: an importable isaacsim in the active python (pip or
# conda install), then the usual python.sh locations.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"

find_python() {
    for py in python python3; do
        if command -v "$py" >/dev/null 2>&1 &&
           "$py" -c "import isaacsim" >/dev/null 2>&1; then
            echo "$py"
            return 0
        fi
    done
    for cand in \
        "$HOME/isaacsim/python.sh" \
        "$HOME/.local/share/ov/pkg"/isaac_sim-*/python.sh \
        "$HOME"/isaac-sim*/python.sh \
        /isaac-sim/python.sh \
        /opt/isaacsim/python.sh; do
        if [ -x "$cand" ]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

PY="$(find_python)" || {
    echo "Could not find Isaac Sim." >&2
    echo "Searched: importable isaacsim in python/python3, then" >&2
    echo "  ~/isaacsim/python.sh, ~/.local/share/ov/pkg/isaac_sim-*/python.sh," >&2
    echo "  ~/isaac-sim*/python.sh, /isaac-sim/python.sh, /opt/isaacsim/python.sh" >&2
    echo "Activate the environment holding Isaac Sim, or pass its python" >&2
    echo "explicitly, for example:" >&2
    echo "  /path/to/python.sh $HERE/run_validation.py --backend isaac ..." >&2
    exit 1
}
echo "Isaac Sim python: $PY"

if ! "$PY" -c "import stable_baselines3" >/dev/null 2>&1; then
    echo "Installing stable-baselines3 and gymnasium into that python..."
    "$PY" -m pip install stable-baselines3 gymnasium || exit 1
fi

cd "$HERE" || exit 1
case "$MODE" in
    smoke)
        "$PY" run_validation.py --backend isaac --maps 2 --n 4 --m 20 \
            --samples 2 --reveal lidar --out results_isaac_smoke2
        ;;
    full)
        "$PY" run_validation.py --backend isaac --maps 10 --n 3,4,5 \
            --m 20,25 --samples 5 --reveal lidar --out results_isaac
        ;;
    noise)
        for s in 0.00 0.05 0.10; do
            "$PY" run_validation.py --backend isaac --maps 10 --n 4 --m 20 \
                --samples 5 --reveal lidar --noise "$s" \
                --out "results_isaac_n$s"
        done
        ;;
    *)
        echo "usage: bash run_smoke.sh [smoke|full|noise]" >&2
        exit 1
        ;;
esac
