"""Pure-Python continuous backend with the same interface as IsaacBackend.

Robots are velocity-limited holonomic point kinematics, obstacles are the
grid cells of the true map, and the LiDAR is an exact segment-vs-cell
raycaster.  Used for (a) unit and parity tests on any machine, and
(b) a continuous-kinematics ablation without Isaac Sim.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


class MockBackend:
    def __init__(self, vmax: float = 0.5, dt: float = 1.0 / 60.0,
                 n_beams: int = 72, max_range_factor: float = 2.5):
        self.vmax = vmax
        self.dt = dt
        self.n_beams = n_beams
        self.max_range_factor = max_range_factor

    # ------------------------------------------------------------- #
    def reset(self, grid: np.ndarray, starts_cells: List[Tuple[int, int]],
              cell_size: float) -> None:
        self.grid = np.asarray(grid, dtype=np.int8)
        self.rows, self.cols = self.grid.shape
        self.cell_size = cell_size
        self.max_range = cell_size * self.max_range_factor
        self.poses = np.array(
            [[(c[0] + 0.5) * cell_size, (c[1] + 0.5) * cell_size]
             for c in starts_cells], dtype=np.float64)

    def get_poses(self) -> np.ndarray:
        return self.poses.copy()

    # ------------------------------------------------------------- #
    def drive_to(self, targets_m: np.ndarray, tol_m: float,
                 max_sim_s: float) -> float:
        """All robots move straight toward their targets at <= vmax."""
        t = 0.0
        targets_m = np.asarray(targets_m, dtype=np.float64)
        while t < max_sim_s:
            delta = targets_m - self.poses
            dist = np.linalg.norm(delta, axis=1)
            if (dist <= tol_m).all():
                break
            step = np.zeros_like(delta)
            moving = dist > tol_m
            step[moving] = (delta[moving].T
                            * np.minimum(self.vmax * self.dt / dist[moving],
                                         1.0)).T
            self.poses = self.poses + step
            t += self.dt
        # snap to targets within tolerance so cell quantisation is exact
        delta = targets_m - self.poses
        dist = np.linalg.norm(delta, axis=1)
        self.poses[dist <= tol_m] = targets_m[dist <= tol_m]
        return t

    # ------------------------------------------------------------- #
    def _cell_is_obstacle(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True                      # world boundary = solid
        return self.grid[r, c] == 1

    def raycast(self, origin_m: np.ndarray) -> np.ndarray:
        """March each beam in small steps until it enters an obstacle cell.

        Returns (n_beams, 3): [hit(0/1), end_x, end_y] where end is the
        first point inside an obstacle cell (hit) or the max-range point.
        """
        out = np.zeros((self.n_beams, 3))
        step = self.cell_size * 0.05
        for b in range(self.n_beams):
            ang = 2.0 * math.pi * b / self.n_beams
            d = np.array([math.cos(ang), math.sin(ang)])
            travelled = step
            hit = False
            end = origin_m + d * self.max_range
            while travelled <= self.max_range:
                p = origin_m + d * travelled
                r = math.floor(p[0] / self.cell_size)
                c = math.floor(p[1] / self.cell_size)
                if self._cell_is_obstacle(int(r), int(c)):
                    hit = True
                    end = p
                    break
                travelled += step
            out[b] = [1.0 if hit else 0.0, end[0], end[1]]
        return out

    def close(self) -> None:
        pass
