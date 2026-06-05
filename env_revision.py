"""
Clean re-implementation of the multi-robot rendezvous environment for the
journal revision. Mirrors the paper's Table III rewards exactly; uses the
gymnasium API; supports SubprocVecEnv vectorisation.

This is a new file. The legacy paper-cited scripts (v*.py, rl_*.py,
map_gen_*.py) are NOT modified.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


REWARD_AREA_DECREASE = +20.0
REWARD_AREA_INCREASE = -0.5
REWARD_COLLISION = -5.0
REWARD_GOAL = +100.0

# Per-robot action lookup: 0=stay, 1=up, 2=down, 3=left, 4=right.
ACTION_DELTAS = np.array([
    [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
], dtype=np.int32)


def _largest_component_size(grid: np.ndarray) -> int:
    m, n = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    best = 0
    for i in range(m):
        for j in range(n):
            if grid[i, j] == 0 and not visited[i, j]:
                stack, count = [(i, j)], 0
                while stack:
                    x, y = stack.pop()
                    if not (0 <= x < m and 0 <= y < n):
                        continue
                    if visited[x, y] or grid[x, y] == 1:
                        continue
                    visited[x, y] = True
                    count += 1
                    stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
                if count > best:
                    best = count
    return best


def _generate_map(rng: np.random.Generator, m: int, n: int,
                  obstacle_density: float) -> np.ndarray:
    target = (m * n) * (1.0 - obstacle_density) * 0.7
    for _ in range(50):
        grid = (rng.random((m, n)) < obstacle_density).astype(np.int8)
        if _largest_component_size(grid) >= target:
            return grid
    return (rng.random((m, n)) < obstacle_density * 0.5).astype(np.int8)


def _bfs_reachable(grid: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    m, n = grid.shape
    reachable = np.zeros_like(grid, dtype=bool)
    stack = [start]
    while stack:
        x, y = stack.pop()
        if not (0 <= x < m and 0 <= y < n):
            continue
        if reachable[x, y] or grid[x, y] == 1:
            continue
        reachable[x, y] = True
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    return reachable


class RendezvousEnv(gym.Env):
    """Multi-robot rendezvous on a grid, paper Table III rewards."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_robots: int,
        map_size: int,
        lidar_radius: int = 1,
        threshold_area: Optional[int] = None,
        max_steps: int = 256,
        obstacle_density: float = 0.20,
    ):
        super().__init__()
        self.n_robots = n_robots
        self.m = self.n = map_size
        self.lidar_radius = lidar_radius
        self.threshold_area = threshold_area if threshold_area is not None else 16
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density

        self.observation_space = spaces.Dict({
            "known_map": spaces.Box(low=-1, high=1, shape=(self.m, self.n), dtype=np.int8),
            "robot_positions": spaces.Box(
                low=0, high=max(self.m, self.n) - 1,
                shape=(n_robots, 2), dtype=np.int32,
            ),
        })
        self.action_space = spaces.MultiDiscrete([5] * n_robots)

        self._true_map: Optional[np.ndarray] = None
        self._known_map: Optional[np.ndarray] = None
        self._positions: Optional[np.ndarray] = None
        self._best_area: float = float("inf")
        self._step_count: int = 0

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._true_map = _generate_map(rng, self.m, self.n, self.obstacle_density)
        free_cells = np.argwhere(self._true_map == 0)
        anchor = tuple(free_cells[rng.integers(0, len(free_cells))])
        reachable = _bfs_reachable(self._true_map, anchor)
        valid = np.argwhere(reachable)
        if len(valid) < self.n_robots:
            # Fallback: regenerate sparser map.
            return self.reset(seed=seed + 1 if seed is not None else None)
        idx = rng.choice(len(valid), size=self.n_robots, replace=False)
        self._positions = valid[idx].astype(np.int32)
        self._known_map = np.full((self.m, self.n), -1, dtype=np.int8)
        self._scan_all()
        self._best_area = self._compute_bounding_area()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        action = np.asarray(action, dtype=np.int32).reshape(self.n_robots)
        proposed = self._positions + ACTION_DELTAS[action]
        reward = 0.0
        any_collision = False

        # Wall and obstacle collisions
        for i in range(self.n_robots):
            x, y = proposed[i]
            if not (0 <= x < self.m and 0 <= y < self.n) or self._true_map[x, y] == 1:
                any_collision = True
                reward += REWARD_COLLISION
                proposed[i] = self._positions[i]

        # Robot-robot collisions (revert both)
        seen = {}
        for i, (x, y) in enumerate(proposed):
            key = (int(x), int(y))
            if key in seen:
                any_collision = True
                reward += REWARD_COLLISION
                proposed[i] = self._positions[i]
                proposed[seen[key]] = self._positions[seen[key]]
            else:
                seen[key] = i

        self._positions = proposed
        self._scan_all()

        current_area = self._compute_bounding_area()
        terminated = False
        if current_area < self._best_area:
            self._best_area = current_area
            if current_area <= self.threshold_area:
                reward += REWARD_GOAL
                terminated = True
            else:
                reward += REWARD_AREA_DECREASE
        else:
            reward += REWARD_AREA_INCREASE

        truncated = self._step_count >= self.max_steps
        info = {
            "bounding_area": current_area,
            "best_area": self._best_area,
            "collision": any_collision,
            "is_success": terminated,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _scan_all(self):
        r = self.lidar_radius
        for x, y in self._positions:
            x0, x1 = max(0, x - r), min(self.m, x + r + 1)
            y0, y1 = max(0, y - r), min(self.n, y + r + 1)
            self._known_map[x0:x1, y0:y1] = self._true_map[x0:x1, y0:y1]

    def _compute_bounding_area(self) -> int:
        xs = self._positions[:, 0]
        ys = self._positions[:, 1]
        return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))

    def _get_obs(self):
        return {
            "known_map": self._known_map.copy(),
            "robot_positions": self._positions.copy(),
        }


def make_env(n_robots: int, map_size: int, seed: int = 0, **kwargs):
    """Factory for SubprocVecEnv: returns a callable that builds an env."""
    def _thunk():
        env = RendezvousEnv(n_robots=n_robots, map_size=map_size, **kwargs)
        env.reset(seed=seed)
        return env
    return _thunk
