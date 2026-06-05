"""
Classical A* rendezvous baseline.

Centralised planner that operates on the same partially-known map the RL
policy sees. At each replanning step it picks a meeting cell on the known
map (centroid of robot positions, snapped to nearest free cell), plans an
A* path for each robot to that cell, and advances every robot one step
along its path. Replans every K steps.

Unknown cells are treated as free (optimistic). Termination matches the
RL env: bounding area at or below threshold.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import heapq
import numpy as np

from env_revision import RendezvousEnv, ACTION_DELTAS


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(known_map: np.ndarray, start, goal) -> Optional[List[Tuple[int, int]]]:
    """A* on a grid; treats -1 (unknown) and 0 (free) as passable, 1 (obstacle) as blocked."""
    m, n = known_map.shape
    if start == goal:
        return [start]
    open_heap = [(manhattan(start, goal), 0, start)]
    came = {start: None}
    g = {start: 0}
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return list(reversed(path))
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < m and 0 <= ny < n):
                continue
            if known_map[nx, ny] == 1:
                continue
            new_g = gc + 1
            if (nx, ny) not in g or new_g < g[(nx, ny)]:
                g[(nx, ny)] = new_g
                came[(nx, ny)] = cur
                heapq.heappush(open_heap, (new_g + manhattan((nx, ny), goal), new_g, (nx, ny)))
    return None


def pick_meeting_cell(known_map: np.ndarray, positions: np.ndarray) -> Tuple[int, int]:
    """Centroid of robot positions, snapped to the nearest non-obstacle cell."""
    cx = int(round(positions[:, 0].mean()))
    cy = int(round(positions[:, 1].mean()))
    m, n = known_map.shape
    cx = max(0, min(m - 1, cx))
    cy = max(0, min(n - 1, cy))
    if known_map[cx, cy] != 1:
        return (cx, cy)
    # BFS for nearest non-obstacle
    best, best_d = (cx, cy), float("inf")
    for r in range(1, max(m, n)):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < m and 0 <= ny < n and known_map[nx, ny] != 1:
                    d = abs(dx) + abs(dy)
                    if d < best_d:
                        best, best_d = (nx, ny), d
        if best_d < float("inf"):
            break
    return best


def delta_to_action(dx: int, dy: int) -> int:
    """Map (dx, dy) to RendezvousEnv action index. 0=stay, 1=up, 2=down, 3=left, 4=right."""
    if dx == 0 and dy == 0: return 0
    if dx == -1 and dy == 0: return 1
    if dx == 1 and dy == 0: return 2
    if dx == 0 and dy == -1: return 3
    if dx == 0 and dy == 1: return 4
    return 0


def run_astar_episode(env: RendezvousEnv, replan_every: int = 5,
                      max_steps: Optional[int] = None) -> dict:
    """Roll the A* baseline on a single env episode. Returns metrics dict."""
    obs, _ = env.reset()
    n_robots = env.n_robots
    total_distance = 0
    per_robot_distance = np.zeros(n_robots, dtype=np.int32)
    steps = 0
    max_steps = max_steps or env.max_steps
    success = False
    paths: List[List[Tuple[int, int]]] = [[] for _ in range(n_robots)]

    while steps < max_steps:
        known = obs["known_map"]
        positions = obs["robot_positions"]
        # Replan
        if steps % replan_every == 0 or any(len(p) <= 1 for p in paths):
            goal = pick_meeting_cell(known, positions)
            paths = []
            for i in range(n_robots):
                p = astar(known, tuple(positions[i]), goal)
                paths.append(p if p else [tuple(positions[i])])

        # Take next step for each robot along its path
        action = np.zeros(n_robots, dtype=np.int32)
        new_paths = []
        for i in range(n_robots):
            if len(paths[i]) <= 1:
                action[i] = 0
                new_paths.append(paths[i])
                continue
            nxt = paths[i][1]
            dx = nxt[0] - positions[i][0]
            dy = nxt[1] - positions[i][1]
            action[i] = delta_to_action(int(dx), int(dy))
            new_paths.append(paths[i][1:])
        paths = new_paths

        obs, reward, terminated, truncated, info = env.step(action)
        # Track distance (only count moves that didn't collide back)
        for i in range(n_robots):
            if action[i] != 0:
                per_robot_distance[i] += 1
                total_distance += 1
        steps += 1
        if terminated:
            success = True
            break
        if truncated:
            break

    return {
        "success": int(success),
        "steps": steps,
        "total_distance": int(total_distance),
        "max_distance": int(per_robot_distance.max()),
        "per_robot_distance": per_robot_distance.tolist(),
    }
