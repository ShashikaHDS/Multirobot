"""Port of the colleague's frontier-based rendezvous algorithm ("Randika
frontier-based", comparison paper) to the paper_rev2 protocol.

Faithful two-phase strategy from the original notebook:

  Phase 1 (rendezvous-biased exploration): every robot repeatedly selects
  the frontier cell (known-free cell adjacent to unknown, searched within
  a window around the fleet's bounding box) that minimises the SUM of
  Manhattan distances to the OTHER robots, and takes one A* step toward
  it (A* over the known map: obstacles and danger cells blocked, unknown
  traversable).  Danger cells are known-free cells within one cell of a
  known obstacle (the original's safety inflation).  Phase 1 ends when
  every robot is within Chebyshev radius R of at least one other robot
  and known-free paths connect robot 0 to every other robot.

  Phase 2 (convergence): BFS distance fields over known, non-danger free
  cells from each robot; the meeting point is the common-reachable cell
  minimising the MAXIMUM distance over robots (minimax), computed once;
  robots then follow known-free A* paths toward it, one cell per step.

Success, termination, and all metrics are judged by the shared
RendezvousEnv exactly as for PPO and the A* pipeline (bounding square
<= threshold obstacle-free, 300-step cap, stochasticity-free: the
algorithm is deterministic given the map seed).

Documented adaptation decisions (for the paper):
  - Sensing is the env's lidar (radius 1) — the same information PPO
    receives; the original used a 7x7 field of view on an 80x80 grid.
  - Geometric parameters scale with the 80->20 map ratio: frontier
    window 20 -> 5, convergence radius 40 -> 10.
  - Our env forbids two robots on one cell (the original allowed
    stacking): a robot whose next cell is held by an arrived robot
    waits; the env's bounding-square success test fires during
    convergence, before exact stacking would be needed.
  - The danger inflation can disconnect the tighter cluster maps used
    here; the evaluation therefore reports the algorithm with and
    without it and the paper quotes the stronger variant.

    python frontier_baseline.py            # evaluate on the 20 held-out maps
"""

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

from env_paper import RendezvousEnv, EnvConfig, UNKNOWN, FREE, OBSTACLE
from astar_paper import astar, _action_for

WINDOW = 5              # frontier search margin around the fleet bbox (80->20 scale)
CONV_RADIUS = 10        # phase-1 convergence radius, Chebyshev (80->20 scale)


# --------------------------------------------------------------------- #
# map annotations                                                       #
# --------------------------------------------------------------------- #
def danger_mask(known: np.ndarray) -> np.ndarray:
    """Known-free cells within one cell (8-neighbourhood) of a known obstacle."""
    R, C = known.shape
    obs = known == OBSTACLE
    near = np.zeros_like(obs)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sx = slice(max(dx, 0), R + min(dx, 0))
            sy = slice(max(dy, 0), C + min(dy, 0))
            tx = slice(max(-dx, 0), R + min(-dx, 0))
            ty = slice(max(-dy, 0), C + min(-dy, 0))
            near[tx, ty] |= obs[sx, sy]
    return near & (known == FREE)


def blocked_mask(known: np.ndarray, use_danger: bool,
                 unknown_blocked: bool) -> np.ndarray:
    b = known == OBSTACLE
    if use_danger:
        b = b | danger_mask(known)
    if unknown_blocked:
        b = b | (known == UNKNOWN)
    return b


def grid_astar(known: np.ndarray, start, goal, use_danger, unknown_blocked):
    """A* wrapper over astar_paper.astar with the original's blocking rules."""
    b = blocked_mask(known, use_danger, unknown_blocked)
    # encode: blocked -> OBSTACLE, everything else FREE for the shared astar
    enc = np.where(b, OBSTACLE, FREE).astype(np.int8)
    return astar(enc, tuple(start), tuple(goal))


# --------------------------------------------------------------------- #
# phase 1: rendezvous-biased frontier exploration                       #
# --------------------------------------------------------------------- #
def find_frontiers(known: np.ndarray, positions, use_danger: bool):
    R, C = known.shape
    x0 = max(0, int(positions[:, 0].min()) - WINDOW)
    x1 = min(R - 1, int(positions[:, 0].max()) + WINDOW)
    y0 = max(0, int(positions[:, 1].min()) - WINDOW)
    y1 = min(C - 1, int(positions[:, 1].max()) + WINDOW)
    dm = danger_mask(known) if use_danger else np.zeros_like(known, dtype=bool)
    out = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            if known[x, y] != FREE or dm[x, y]:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < R and 0 <= ny < C and known[nx, ny] == UNKNOWN:
                    out.append((x, y))
                    break
    return out


def select_frontier(i, positions, frontiers):
    """The original's rule: minimise the summed Manhattan distance to the
    OTHER robots — frontier exploration biased toward the teammates."""
    others = [tuple(p) for j, p in enumerate(positions) if j != i]
    return min(frontiers,
               key=lambda f: sum(abs(f[0] - o[0]) + abs(f[1] - o[1])
                                 for o in others))


def phase1_done(known, positions, use_danger):
    n = len(positions)
    for i in range(n):
        if not any(max(abs(positions[i][0] - positions[j][0]),
                       abs(positions[i][1] - positions[j][1])) <= CONV_RADIUS
                   for j in range(n) if j != i):
            return False
    for j in range(1, n):
        if grid_astar(known, positions[0], positions[j],
                      use_danger, unknown_blocked=True) is None:
            return False
    return True


# --------------------------------------------------------------------- #
# phase 2: minimax meeting point on the known map                       #
# --------------------------------------------------------------------- #
def bfs_field(passable: np.ndarray, start):
    R, C = passable.shape
    INF = np.iinfo(np.int32).max
    dist = np.full((R, C), INF, dtype=np.int32)
    if not passable[tuple(start)]:
        return dist
    dist[tuple(start)] = 0
    dq = deque([tuple(start)])
    while dq:
        x, y = dq.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < R and 0 <= ny < C and passable[nx, ny] \
                    and dist[nx, ny] == INF:
                dist[nx, ny] = dist[x, y] + 1
                dq.append((nx, ny))
    return dist


def minimax_meeting_point(known, positions, use_danger):
    passable = ~blocked_mask(known, use_danger, unknown_blocked=True)
    passable &= known == FREE
    fields = np.stack([bfs_field(passable, p) for p in positions])
    INF = np.iinfo(np.int32).max
    common = (fields < INF).all(axis=0)
    if not common.any():
        return None
    worst = fields.max(axis=0).astype(np.int64)
    worst[~common] = np.iinfo(np.int64).max
    return tuple(int(v) for v in np.unravel_index(np.argmin(worst),
                                                  worst.shape))


# --------------------------------------------------------------------- #
# episode driver                                                        #
# --------------------------------------------------------------------- #
def run_frontier_episode(env: RendezvousEnv, seed=None, use_danger=True):
    obs, _ = env.reset(seed=seed)
    n = env.cfg.num_robots
    phase = 1
    meeting = None
    terminated = truncated = False
    info = {}
    n_obs_col = n_rob_col = 0

    while not (terminated or truncated):
        known = obs["known_map"]
        pos = obs["robot_positions"]

        if phase == 1 and phase1_done(known, pos, use_danger):
            phase = 2
        if phase == 2 and meeting is None:
            meeting = minimax_meeting_point(known, pos, use_danger)
            if meeting is None:
                phase = 1          # not yet connected enough; keep exploring

        targets = []
        if phase == 1:
            frontiers = find_frontiers(known, pos, use_danger)
            for i in range(n):
                targets.append(select_frontier(i, pos, frontiers)
                               if frontiers else None)
        else:
            targets = [meeting] * n

        arrived = [targets[i] is None or tuple(pos[i]) == targets[i]
                   for i in range(n)]
        parked = {tuple(pos[i]) for i in range(n) if arrived[i]}
        actions = []
        for i in range(n):
            if arrived[i]:
                actions.append(4)
                continue
            path = grid_astar(known, tuple(pos[i]), targets[i], use_danger,
                              unknown_blocked=(phase == 2))
            if path is None or len(path) <= 1:
                actions.append(4)
                continue
            nxt = path[1]
            if nxt in parked:
                actions.append(4)
            else:
                cur = tuple(pos[i])
                actions.append(_action_for((nxt[0] - cur[0], nxt[1] - cur[1])))

        obs, r, terminated, truncated, info = env.step(actions)
        n_obs_col += info["obstacle_collisions"]
        n_rob_col += info["robot_collisions"]

    d = env.distances.astype(float)
    jain = float((d.sum() ** 2) / (len(d) * (d ** 2).sum())) if d.sum() > 0 else 1.0
    ok = bool(info.get("is_success", False))
    return {
        "success": ok,
        "cf_success": ok and n_obs_col == 0,
        "steps": env.step_count,
        "total_distance": info.get("total_distance", int(env.distances.sum())),
        "max_distance": info.get("max_distance", int(env.distances.max())),
        "jain": jain,
        "obs_collisions": n_obs_col,
        "robot_collisions": n_rob_col,
        "per_robot_distances": [int(v) for v in env.distances],
    }


# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", type=str, default="results_frontier")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [(2, 20), (3, 20), (4, 20), (5, 20), (5, 25)]
    fields = ["config", "n", "m", "variant", "episode", "success",
              "cf_success", "steps", "total_distance", "max_distance",
              "jain", "obs_collisions", "robot_collisions",
              "per_robot_distances"]
    rows = []
    for (n, m) in configs:
        env = RendezvousEnv(EnvConfig(num_robots=n, rows=m, cols=m))
        for variant, use_danger in (("danger", True), ("no_danger", False)):
            for ep in range(args.episodes):
                met = dict(run_frontier_episode(env, seed=10_000 + ep,
                                                use_danger=use_danger))
                met["per_robot_distances"] = json.dumps(
                    met["per_robot_distances"])
                rows.append({"config": f"N{n}_M{m}", "n": n, "m": m,
                             "variant": variant, "episode": ep, **met})
        env.close()
        for variant in ("danger", "no_danger"):
            sel = [r for r in rows if r["config"] == f"N{n}_M{m}"
                   and r["variant"] == variant]
            print(f"N{n}_M{m} [{variant}]: success "
                  f"{np.mean([r['success'] for r in sel]):.2f}  "
                  f"steps {np.mean([r['steps'] for r in sel]):.0f}  "
                  f"dist {np.mean([r['total_distance'] for r in sel]):.0f}  "
                  f"jain {np.mean([r['jain'] for r in sel]):.3f}", flush=True)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_dir/'results.csv'}")


if __name__ == "__main__":
    main()
