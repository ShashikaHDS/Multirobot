"""
Generalized multi-robot rendezvous environment (v8).

Improvements over v7:
- Variable robot count per episode (cfg.min_robots..cfg.max_robots) with padding +
  presence mask, so a single trained policy generalizes to any count in range.
- Decoupled robot footprint from grid resolution: the planning grid is `k` times
  finer than the robot. k=1 reproduces the paper.
- CNN-friendly observation dict:
    global: (4, coarse, coarse)        - downsampled view of the whole world
    local : (max_robots, 5, crop, crop) - ego-centric crops per robot slot
    robot_xy: (max_robots, 2)           - normalized [0,1] positions
    presence: (max_robots,)             - 1 if active, 0 if padding
  Channel layout: known_free, known_obs, unknown, self_pos, other_pos.
- Reward values exposed via cfg.rewards (paper Table III defaults), so
  reward_sensitivity.py can sweep them without editing this file.

State conventions:
- fine_map[x, y] in {0,1}  : terrain obstacles at fine resolution
- known_map[x, y] in {0,1,2}: 0=unobserved, 1=known free, 2=known obstacle
- robot_xy[i] = (x, y)      : top-left of robot i's k x k footprint, fine coords
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym


PAPER_REWARDS = {
    "area_decrease":   20.0,
    "area_increase":   -0.5,
    "collide_obstacle": -5.0,
    "collide_robot":   -5.0,
    "goal":            100.0,
    "step":            0.0,
}


@dataclass
class EnvConfig:
    map_rows: int = 20
    map_cols: int = 20
    cells_per_robot: int = 2
    step_size_cells: int = 1
    # max_robots = number of padding slots; defines the obs/action shape and
    # MUST stay constant across curriculum stages so a single policy can carry
    # weights between stages.
    max_robots: int = 6
    min_robots: int = 2
    # max_active = upper bound on randomly chosen ACTIVE robot count per
    # episode; can vary per stage. Defaults to max_robots if None.
    max_active: Optional[int] = None
    field_of_view_cells: int = 3
    coarse_size: int = 32
    crop_size: int = 21
    goal_side_logical: int = 4
    max_steps: int = 500
    rewards: dict = field(default_factory=lambda: dict(PAPER_REWARDS))
    num_clusters: int = 5
    cluster_size_range: tuple = (2, 10)
    min_distance_between_clusters: int = 3


class MapGen:
    @staticmethod
    def generate(rows, cols, num_clusters, cluster_size_range, min_distance, rng):
        grid = np.zeros((rows, cols), dtype=np.uint8)
        centers = []
        attempts = 0
        while len(centers) < num_clusters and attempts < num_clusters * 50:
            attempts += 1
            cx, cy = rng.randint(0, rows), rng.randint(0, cols)
            if all(np.hypot(cx - x, cy - y) >= min_distance for x, y in centers):
                centers.append((cx, cy))
        for cx, cy in centers:
            target = rng.randint(cluster_size_range[0], cluster_size_range[1] + 1)
            cells = [(cx, cy)]
            grid[cx, cy] = 1
            tries = 0
            while len(cells) < target and tries < target * 10:
                tries += 1
                px, py = cells[rng.randint(0, len(cells))]
                dx, dy = [(0, 1), (0, -1), (1, 0), (-1, 0)][rng.randint(0, 4)]
                nx, ny = px + dx, py + dy
                if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                    grid[nx, ny] = 1
                    cells.append((nx, ny))
        return grid


class GeneralizedRendezvousEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.RandomState()
        self._screen = None
        self._cell_px = 0

        c = self.cfg
        self.k = c.cells_per_robot
        self.fine_rows = c.map_rows * self.k
        self.fine_cols = c.map_cols * self.k

        self.action_space = gym.spaces.MultiDiscrete([5] * c.max_robots)
        self.observation_space = gym.spaces.Dict({
            "global":   gym.spaces.Box(0.0, 1.0, shape=(4, c.coarse_size, c.coarse_size), dtype=np.float32),
            "local":    gym.spaces.Box(0.0, 1.0, shape=(c.max_robots, 5, c.crop_size, c.crop_size), dtype=np.float32),
            "robot_xy": gym.spaces.Box(0.0, 1.0, shape=(c.max_robots, 2), dtype=np.float32),
            "presence": gym.spaces.Box(0.0, 1.0, shape=(c.max_robots,), dtype=np.float32),
        })

        self.fine_map = None
        self.known_map = None
        self.robot_xy = None
        self.presence = None
        self.num_active = 0
        self.step_count = 0
        self.min_square_area = None
        self.distances = None

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        return [seed]

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        c = self.cfg
        self.step_count = 0
        upper = c.max_active if c.max_active is not None else c.max_robots
        upper = min(upper, c.max_robots)
        self.num_active = self.rng.randint(c.min_robots, upper + 1)
        self.presence = np.zeros(c.max_robots, dtype=np.float32)
        self.presence[:self.num_active] = 1.0

        logical = MapGen.generate(
            c.map_rows, c.map_cols,
            c.num_clusters, c.cluster_size_range,
            c.min_distance_between_clusters, self.rng,
        )
        self.fine_map = np.kron(logical, np.ones((self.k, self.k), dtype=np.uint8))
        self.known_map = np.zeros_like(self.fine_map, dtype=np.uint8)

        self.robot_xy = np.zeros((c.max_robots, 2), dtype=np.int32)
        placed = []
        attempts = 0
        while len(placed) < self.num_active and attempts < 5000:
            attempts += 1
            rx = self.rng.randint(0, self.fine_rows - self.k + 1)
            ry = self.rng.randint(0, self.fine_cols - self.k + 1)
            if self.fine_map[rx:rx + self.k, ry:ry + self.k].any():
                continue
            if any(abs(rx - px) < self.k and abs(ry - py) < self.k for px, py in placed):
                continue
            placed.append((rx, ry))
        if len(placed) < self.num_active:
            # extremely cluttered map: fall back to fewer robots rather than crash
            self.num_active = len(placed)
            self.presence[:] = 0.0
            self.presence[:self.num_active] = 1.0
        for i, (rx, ry) in enumerate(placed):
            self.robot_xy[i] = (rx, ry)

        for i in range(self.num_active):
            self._update_known_at(self.robot_xy[i])

        self.distances = np.zeros(c.max_robots, dtype=np.float32)
        self.min_square_area, _, _ = self._compute_square()
        return self._observation(), {}

    # ------------------------------------------------------------------ step
    def step(self, action):
        c = self.cfg
        self.step_count += 1
        rewards_cfg = c.rewards
        reward = rewards_cfg.get("step", 0.0)

        active = list(range(self.num_active))
        candidate = self.robot_xy.copy()
        collide_obs = 0
        collide_rob = 0

        for i in active:
            a = int(action[i])
            proposed = self._move(self.robot_xy[i], a)
            if self._footprint_hits_obstacle(proposed):
                collide_obs += 1
                continue
            blocked = False
            for j in active:
                if j == i:
                    continue
                if (abs(proposed[0] - candidate[j, 0]) < self.k
                        and abs(proposed[1] - candidate[j, 1]) < self.k):
                    blocked = True
                    break
            if blocked:
                collide_rob += 1
                continue
            self.distances[i] += float(np.abs(proposed - self.robot_xy[i]).sum())
            candidate[i] = proposed

        self.robot_xy = candidate
        for i in active:
            self._update_known_at(self.robot_xy[i])

        if collide_obs:
            reward += rewards_cfg["collide_obstacle"] * collide_obs
        if collide_rob:
            reward += rewards_cfg["collide_robot"] * collide_rob

        new_area, free, _ = self._compute_square()
        goal_area_fine = (c.goal_side_logical * self.k) ** 2

        success = bool(new_area <= goal_area_fine and free)
        terminated = success
        if success:
            reward += rewards_cfg["goal"]
        elif new_area < self.min_square_area:
            reward += rewards_cfg["area_decrease"]
        elif new_area > self.min_square_area:
            reward += rewards_cfg["area_increase"]
        self.min_square_area = min(self.min_square_area, new_area)

        truncated = (not terminated) and self.step_count >= c.max_steps

        info = {
            "collide_obstacle": collide_obs,
            "collide_robot":   collide_rob,
            "min_square_area": float(self.min_square_area),
            "current_square_area": float(new_area),
            "num_active":      int(self.num_active),
            "distances":       self.distances.copy(),
            "success":         success,
            "steps":           self.step_count,
        }
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ helpers
    def _move(self, pos, a):
        s = self.cfg.step_size_cells
        x, y = int(pos[0]), int(pos[1])
        if a == 0:
            x = max(0, x - s)
        elif a == 1:
            x = min(self.fine_rows - self.k, x + s)
        elif a == 2:
            y = max(0, y - s)
        elif a == 3:
            y = min(self.fine_cols - self.k, y + s)
        return np.array([x, y], dtype=np.int32)

    def _footprint_hits_obstacle(self, pos):
        x, y = int(pos[0]), int(pos[1])
        return bool(self.fine_map[x:x + self.k, y:y + self.k].any())

    def _update_known_at(self, pos):
        x, y = int(pos[0]), int(pos[1])
        f = self.cfg.field_of_view_cells
        x0 = max(0, x - f)
        x1 = min(self.fine_rows, x + self.k + f)
        y0 = max(0, y - f)
        y1 = min(self.fine_cols, y + self.k + f)
        block = self.fine_map[x0:x1, y0:y1]
        view = self.known_map[x0:x1, y0:y1]
        view[block == 0] = 1
        view[block == 1] = 2

    def _compute_square(self):
        active = self.robot_xy[:self.num_active]
        if active.shape[0] == 0:
            return 0, True, (0, 0, 0, 0)
        k = self.k
        xs_lo = int(active[:, 0].min())
        xs_hi = int(active[:, 0].max()) + k - 1
        ys_lo = int(active[:, 1].min())
        ys_hi = int(active[:, 1].max()) + k - 1
        side = max(xs_hi - xs_lo, ys_hi - ys_lo) + 1
        x_max = xs_lo + side - 1
        y_max = ys_lo + side - 1
        free = (x_max < self.fine_rows and y_max < self.fine_cols
                and not self.fine_map[xs_lo:x_max + 1, ys_lo:y_max + 1].any())
        return side * side, free, (xs_lo, x_max, ys_lo, y_max)

    # ------------------------------------------------------------------ observation
    def _observation(self):
        c = self.cfg
        kf = (self.known_map == 1).astype(np.float32)
        ko = (self.known_map == 2).astype(np.float32)
        ku = (self.known_map == 0).astype(np.float32)
        rb = np.zeros_like(kf)
        for i in range(self.num_active):
            x, y = self.robot_xy[i]
            rb[x:x + self.k, y:y + self.k] = 1.0
        global_stack = np.stack([kf, ko, ku, rb], axis=0)
        global_obs = self._block_pool(global_stack, c.coarse_size)

        local = self._all_local_crops(kf, ko, ku)

        rxy = np.zeros((c.max_robots, 2), dtype=np.float32)
        for i in range(self.num_active):
            rxy[i, 0] = self.robot_xy[i, 0] / max(1, self.fine_rows - 1)
            rxy[i, 1] = self.robot_xy[i, 1] / max(1, self.fine_cols - 1)

        return {
            "global":   global_obs.astype(np.float32),
            "local":    local.astype(np.float32),
            "robot_xy": rxy,
            "presence": self.presence.astype(np.float32),
        }

    def _block_pool(self, stack, target):
        ch, h, w = stack.shape
        if h == target and w == target:
            return stack
        # Upsample if smaller than target
        if h < target or w < target:
            zh = max(1, -(-target // h))
            zw = max(1, -(-target // w))
            up = np.repeat(np.repeat(stack, zh, axis=1), zw, axis=2)
            return up[:, :target, :target]
        # Downsample by block-mean (pad to multiple of target first)
        ph = (target - h % target) % target
        pw = (target - w % target) % target
        if ph or pw:
            stack = np.pad(stack, ((0, 0), (0, ph), (0, pw)), mode="edge")
        h2, w2 = stack.shape[1], stack.shape[2]
        bh, bw = h2 // target, w2 // target
        return stack.reshape(ch, target, bh, target, bw).mean(axis=(2, 4))

    def _all_local_crops(self, kf_full, ko_full, ku_full):
        c = self.cfg
        cs = c.crop_size
        half = cs // 2
        pad = half + self.k

        all_robots = np.zeros_like(kf_full)
        for i in range(self.num_active):
            x, y = self.robot_xy[i]
            all_robots[x:x + self.k, y:y + self.k] = 1.0

        pkf = np.pad(kf_full, pad, mode="constant", constant_values=0)
        pko = np.pad(ko_full, pad, mode="constant", constant_values=1)
        pku = np.pad(ku_full, pad, mode="constant", constant_values=0)
        pall = np.pad(all_robots, pad, mode="constant", constant_values=0)

        out = np.zeros((c.max_robots, 5, cs, cs), dtype=np.float32)
        for i in range(self.num_active):
            cx = int(self.robot_xy[i, 0]) + self.k // 2
            cy = int(self.robot_xy[i, 1]) + self.k // 2
            sx = cx + pad - half
            sy = cy + pad - half
            out[i, 0] = pkf[sx:sx + cs, sy:sy + cs]
            out[i, 1] = pko[sx:sx + cs, sy:sy + cs]
            out[i, 2] = pku[sx:sx + cs, sy:sy + cs]
            kk = self.k
            offset = half - kk // 2
            out[i, 3, offset:offset + kk, offset:offset + kk] = 1.0
            other = pall[sx:sx + cs, sy:sy + cs] - out[i, 3]
            out[i, 4] = np.clip(other, 0.0, 1.0)
        return out

    # ------------------------------------------------------------------ render
    def render(self, mode="human"):
        if mode != "human":
            return
        import pygame as pg
        if self._screen is None:
            pg.init()
            self._cell_px = max(2, 600 // max(self.fine_rows, self.fine_cols))
            self._screen = pg.display.set_mode(
                (self.fine_cols * self._cell_px, self.fine_rows * self._cell_px))
            pg.display.set_caption("v8 generalized rendezvous")
        cs = self._cell_px
        s = self._screen
        s.fill((255, 255, 255))
        for x in range(self.fine_rows):
            for y in range(self.fine_cols):
                rect = pg.Rect(y * cs, x * cs, cs, cs)
                if self.fine_map[x, y] == 1:
                    pg.draw.rect(s, (0, 0, 0), rect)
                elif self.known_map[x, y] == 1:
                    pg.draw.rect(s, (255, 250, 200), rect)
                else:
                    pg.draw.rect(s, (180, 180, 220), rect)
        for i in range(self.num_active):
            x, y = self.robot_xy[i]
            pg.draw.rect(s, (255, 0, 0),
                         pg.Rect(y * cs, x * cs, self.k * cs, self.k * cs))
        _, _, b = self._compute_square()
        x0, x1, y0, y1 = b
        pg.draw.rect(s, (0, 200, 0),
                     (y0 * cs, x0 * cs, (y1 - y0 + 1) * cs, (x1 - x0 + 1) * cs), 2)
        pg.event.pump()
        pg.display.update()

    def close(self):
        if self._screen is not None:
            import pygame as pg
            pg.quit()
            self._screen = None


if __name__ == "__main__":
    env = GeneralizedRendezvousEnv()
    obs, _ = env.reset(seed=0)
    print("Observation shapes:")
    for k, v in obs.items():
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")
    print(f"num_active={env.num_active}")
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            print("episode end:", info)
            break
    env.close()
