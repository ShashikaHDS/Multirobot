"""A* rendezvous baseline with the THREE meeting-point heuristics the
manuscript claims, operating on the same partially observed map and the
same per-step cadence as the RL policy.

Heuristics (Section IV-H of the manuscript):
  "median"    geometric median of robot positions (Weiszfeld), snapped to
              the nearest cell not known to be an obstacle
  "component" centroid of the largest 4-connected component of *known
              free* cells, snapped likewise
  "bbox"      centre of the current fleet bounding box, snapped likewise

Planning: 4-connected A* with Manhattan heuristic on the known map,
treating UNKNOWN cells as free (optimistic).  Re-plans every
`replan_every` steps (K=5 in the paper) or when a robot exhausts its path.
Termination/metrics come from the shared env, so success semantics are
identical to PPO's.
"""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from env_paper import RendezvousEnv, UNKNOWN, FREE, OBSTACLE

HEURISTICS = ("median", "component", "bbox")


# --------------------------------------------------------------------- #
# meeting-point heuristics                                              #
# --------------------------------------------------------------------- #
def _snap_to_passable(known: np.ndarray, cell: Tuple[int, int]) -> Tuple[int, int]:
    """Nearest (Manhattan ring scan) cell not known to be an obstacle."""
    R, C = known.shape
    cx = int(np.clip(cell[0], 0, R - 1))
    cy = int(np.clip(cell[1], 0, C - 1))
    if known[cx, cy] != OBSTACLE:
        return (cx, cy)
    for r in range(1, R + C):
        best = None
        for dx in range(-r, r + 1):
            dy_abs = r - abs(dx)
            for dy in ({dy_abs, -dy_abs}):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < R and 0 <= ny < C and known[nx, ny] != OBSTACLE:
                    if best is None:
                        best = (nx, ny)
        if best is not None:
            return best
    return (cx, cy)


def meeting_point(known: np.ndarray, positions: np.ndarray,
                  heuristic: str) -> Tuple[int, int]:
    if heuristic == "median":
        pts = positions.astype(float)
        m = pts.mean(axis=0)
        for _ in range(50):                       # Weiszfeld iterations
            d = np.linalg.norm(pts - m, axis=1)
            if (d < 1e-9).any():
                break
            w = 1.0 / np.maximum(d, 1e-9)
            m_new = (pts * w[:, None]).sum(axis=0) / w.sum()
            if np.linalg.norm(m_new - m) < 1e-6:
                m = m_new
                break
            m = m_new
        cand = (int(round(m[0])), int(round(m[1])))
    elif heuristic == "component":
        R, C = known.shape
        seen = np.zeros_like(known, dtype=bool)
        best_comp: List[Tuple[int, int]] = []
        for sx in range(R):
            for sy in range(C):
                if known[sx, sy] == FREE and not seen[sx, sy]:
                    comp = []
                    stack = [(sx, sy)]
                    seen[sx, sy] = True
                    while stack:
                        x, y = stack.pop()
                        comp.append((x, y))
                        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < R and 0 <= ny < C and not seen[nx, ny] \
                                    and known[nx, ny] == FREE:
                                seen[nx, ny] = True
                                stack.append((nx, ny))
                    if len(comp) > len(best_comp):
                        best_comp = comp
        if best_comp:
            arr = np.array(best_comp)
            cand = (int(round(arr[:, 0].mean())), int(round(arr[:, 1].mean())))
        else:                                      # nothing known free yet
            cand = (int(round(positions[:, 0].mean())),
                    int(round(positions[:, 1].mean())))
    elif heuristic == "bbox":
        cand = (int(round((positions[:, 0].min() + positions[:, 0].max()) / 2)),
                int(round((positions[:, 1].min() + positions[:, 1].max()) / 2)))
    else:
        raise ValueError(heuristic)
    return _snap_to_passable(known, cand)


# --------------------------------------------------------------------- #
# A* on the known map (unknown optimistic-free)                         #
# --------------------------------------------------------------------- #
def astar(known: np.ndarray, start: Tuple[int, int],
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]
    R, C = known.shape
    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    open_heap = [(h(start), 0, start)]
    g: Dict[Tuple[int, int], int] = {start: 0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    while open_heap:
        f, gc, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if gc > g.get(cur, 1 << 30):
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < R and 0 <= nxt[1] < C):
                continue
            if known[nxt] == OBSTACLE:
                continue                      # unknown (-1) passes: optimistic
            ng = gc + 1
            if ng < g.get(nxt, 1 << 30):
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def _action_for(delta: Tuple[int, int]) -> int:
    return {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3, (0, 0): 4}[delta]


# --------------------------------------------------------------------- #
# episode driver                                                        #
# --------------------------------------------------------------------- #
def run_astar_episode(env: RendezvousEnv, heuristic: str,
                      replan_every: int = 5, seed: Optional[int] = None):
    """Roll one episode with the A* controller. Returns the metrics dict."""
    obs, _ = env.reset(seed=seed)
    n = env.cfg.num_robots
    paths: List[Optional[List[Tuple[int, int]]]] = [None] * n
    steps_since_plan = replan_every            # force plan on first step
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        known = obs["known_map"]
        pos = obs["robot_positions"]
        need = steps_since_plan >= replan_every or any(
            paths[i] is None or len(paths[i]) <= 1 for i in range(n))
        if need:
            goal = meeting_point(known, pos, heuristic)
            paths = [astar(known, tuple(pos[i]), goal) for i in range(n)]
            steps_since_plan = 0
        actions = []
        for i in range(n):
            if paths[i] is None or len(paths[i]) <= 1:
                actions.append(4)              # stay
            else:
                cur, nxt = paths[i][0], paths[i][1]
                actions.append(_action_for((nxt[0] - cur[0], nxt[1] - cur[1])))
        obs, r, terminated, truncated, info = env.step(actions)
        steps_since_plan += 1
        # advance paths for robots whose move was realised
        for i in range(n):
            if paths[i] is not None and len(paths[i]) > 1 \
                    and tuple(obs["robot_positions"][i]) == paths[i][1]:
                paths[i] = paths[i][1:]
    return {
        "success": bool(info.get("is_success", False)),
        "steps": env.step_count,
        "total_distance": info.get("total_distance", int(env.distances.sum())),
        "max_distance": info.get("max_distance", int(env.distances.max())),
    }
