"""A* rendezvous baseline with the THREE meeting-point heuristics the
manuscript claims, operating on the same partially observed map and the
same per-step cadence as the RL policy.

Heuristics (Section IV-H of the manuscript):
  "median"    geometric median of robot positions (Weiszfeld), snapped to
              the nearest *goal-valid* cell
  "component" centroid of the largest 4-connected component of *known
              free* cells, snapped likewise
  "bbox"      centre of the current fleet bounding box, snapped likewise

Goal validity (steel-manning): a meeting cell is only accepted if some
t x t window containing it (t = floor(sqrt(threshold_area)), i.e. the
4x4 success box) lies fully inside the map and contains no *known*
obstacle -- unknown cells count as free (optimistic).  Without this
check the fleet can gather at a point whose anchored bounding square can
never satisfy the env's success predicate (flush against a border, or
with an obstacle inside) and freeze there until truncation.

Planning: 4-connected A* with Manhattan heuristic on the known map,
treating UNKNOWN cells as free (optimistic).  Re-plans every
`replan_every` steps (K=5 in the paper).  If the fleet makes no spatial
progress for `stall_patience` consecutive steps, the current meeting
cell is blacklisted for the episode and a new one is selected -- the
optimistic goal turned out to be invalid on the true map.

Execution details: robots follow their planned paths one cell per step;
a robot whose next cell is occupied by an already-arrived (parked) robot
waits instead of bumping into it, so the baseline is not charged
spurious collision penalties.  Termination and all metrics come from the
shared env, so success semantics are identical to PPO's.
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from env_paper import RendezvousEnv, UNKNOWN, FREE, OBSTACLE

HEURISTICS = ("median", "component", "bbox")


# --------------------------------------------------------------------- #
# goal validity                                                         #
# --------------------------------------------------------------------- #
def goal_valid_cells(known: np.ndarray, t: int) -> np.ndarray:
    """Boolean mask of cells contained in at least one t x t window that is
    fully in-bounds and free of KNOWN obstacles (unknown = optimistic)."""
    R, C = known.shape
    if R < t or C < t:
        return np.zeros((R, C), dtype=bool)
    obs = (known == OBSTACLE).astype(np.int32)
    ii = np.zeros((R + 1, C + 1), dtype=np.int32)
    ii[1:, 1:] = obs.cumsum(0).cumsum(1)
    win = ii[t:, t:] - ii[:-t, t:] - ii[t:, :-t] + ii[:-t, :-t]   # obstacle counts
    ok = np.zeros((R, C), dtype=bool)
    for a, b in zip(*np.nonzero(win == 0)):
        ok[a:a + t, b:b + t] = True
    return ok


def _snap_to_valid(known: np.ndarray, cell: Tuple[int, int], t: int,
                   blacklist: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Nearest (Manhattan ring scan) goal-valid, non-blacklisted cell."""
    R, C = known.shape
    valid = goal_valid_cells(known, t)
    cx = int(np.clip(cell[0], 0, R - 1))
    cy = int(np.clip(cell[1], 0, C - 1))

    def ok(x, y):
        return valid[x, y] and (x, y) not in blacklist

    if ok(cx, cy):
        return (cx, cy)
    for r in range(1, R + C):
        for dx in range(-r, r + 1):
            dy_abs = r - abs(dx)
            for dy in ({dy_abs, -dy_abs}):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < R and 0 <= ny < C and ok(nx, ny):
                    return (nx, ny)
    # nothing valid (extremely degenerate) -- fall back to any non-obstacle
    for r in range(0, R + C):
        for dx in range(-r, r + 1):
            dy_abs = r - abs(dx)
            for dy in ({dy_abs, -dy_abs}):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < R and 0 <= ny < C and known[nx, ny] != OBSTACLE:
                    return (nx, ny)
    return (cx, cy)


def meeting_point(known: np.ndarray, positions: np.ndarray, heuristic: str,
                  t: int, blacklist: Set[Tuple[int, int]]) -> Tuple[int, int]:
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
    return _snap_to_valid(known, cand, t, blacklist)


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
                      replan_every: int = 5, seed: Optional[int] = None,
                      stall_patience: Optional[int] = None):
    """Roll one episode with the A* controller. Returns the metrics dict."""
    obs, _ = env.reset(seed=seed)
    n = env.cfg.num_robots
    t = max(2, int(np.floor(np.sqrt(env.cfg.threshold_area))))
    stall_patience = stall_patience or 2 * replan_every
    blacklist: Set[Tuple[int, int]] = set()
    paths: List[Optional[List[Tuple[int, int]]]] = [None] * n
    goal: Optional[Tuple[int, int]] = None
    steps_since_plan = replan_every            # force a plan on the first step
    stall = 0
    last_positions: Optional[np.ndarray] = None
    terminated = truncated = False
    info = {}
    n_obs_col = n_rob_col = 0

    while not (terminated or truncated):
        known = obs["known_map"]
        pos = obs["robot_positions"]

        # K-step replanning cadence (plus forced replans after a stall);
        # arrived robots do NOT trigger extra replans (that would make the
        # effective cadence per-step, contradicting the K=5 protocol)
        if goal is None or steps_since_plan >= replan_every:
            goal = meeting_point(known, pos, heuristic, t, blacklist)
            paths = [astar(known, tuple(pos[i]), goal) for i in range(n)]
            steps_since_plan = 0

        arrived = [tuple(pos[i]) == goal or
                   (paths[i] is not None and len(paths[i]) <= 1)
                   for i in range(n)]
        parked = {tuple(pos[i]) for i in range(n) if arrived[i]}
        actions = []
        for i in range(n):
            if arrived[i] or paths[i] is None or len(paths[i]) <= 1:
                actions.append(4)              # stay
            else:
                cur_i, nxt = tuple(pos[i]), paths[i][1]
                if nxt in parked:
                    actions.append(4)          # wait, don't bump a parked robot
                else:
                    actions.append(_action_for((nxt[0] - cur_i[0],
                                                nxt[1] - cur_i[1])))

        obs, r, terminated, truncated, info = env.step(actions)
        n_obs_col += info["obstacle_collisions"]
        n_rob_col += info["robot_collisions"]
        steps_since_plan += 1
        new_pos = obs["robot_positions"]

        # advance path pointers for realized moves
        for i in range(n):
            if paths[i] is not None and len(paths[i]) > 1 \
                    and tuple(new_pos[i]) == paths[i][1]:
                paths[i] = paths[i][1:]

        # stall detection: no robot moved for stall_patience steps -> the
        # optimistic goal is invalid on the true map; blacklist and re-target
        if last_positions is not None and np.array_equal(new_pos, last_positions):
            stall += 1
            if stall >= stall_patience and goal is not None:
                blacklist.add(goal)
                goal = None                    # force replan with blacklist
                stall = 0
        else:
            stall = 0
        last_positions = new_pos.copy()

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
    }
