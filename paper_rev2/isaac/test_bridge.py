"""Tests for the continuous-space bridge (run: python test_bridge.py).

The headline test is PARITY: with the mock backend, oracle 'grid' reveal,
zero noise, and exact waypoint arrival, the continuous pipeline must
reproduce a plain env_paper rollout step for step, since the policy then
sees byte-identical observations at every step.  If parity holds, any
difference measured under 'lidar' reveal or pose noise is attributable to
those factors alone rather than to bridge bugs.

Plain asserts, no pytest required (pytest collects them too).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

from env_paper import RendezvousEnv, EnvConfig, FREE, OBSTACLE, UNKNOWN  # noqa: E402
from bridge import (LockstepRunner, RunnerConfig, OccupancyMapper,  # noqa: E402
                    cell_center, pose_to_cell, resolve_conflicts,
                    bounding_square, square_free)
from mock_env import MockBackend  # noqa: E402


def test_cell_pose_roundtrip():
    cs = 0.25
    for cell in [(0, 0), (3, 7), (19, 19)]:
        p = cell_center(cell, cs)
        assert pose_to_cell(p, cs, 20, 20) == cell, cell
    print("PASS cell_pose_roundtrip")


def test_conflicts_delegate_to_env():
    cur = [(0, 0), (0, 2)]
    targets = [(0, 1), (0, 1)]
    flags = resolve_conflicts(cur, targets)
    assert targets == [(0, 0), (0, 2)], "same-target not reverted"
    assert all(flags)
    cur = [(0, 0), (0, 1)]
    targets = [(0, 1), (0, 0)]
    flags = resolve_conflicts(cur, targets)
    assert targets == [(0, 0), (0, 1)], "swap not reverted"
    print("PASS conflicts_delegate_to_env")


def test_grid_mode_matches_env_reveal():
    """OccupancyMapper 'grid' mode == env_paper._reveal, cell for cell."""
    env = RendezvousEnv(EnvConfig(num_robots=4))
    env.reset(seed=4242)
    m = OccupancyMapper(env.cfg.rows, env.cfg.cols, 0.25, "grid", 1)
    for p in env.positions:
        m.update_grid_mode(env.grid_map, tuple(p))
    assert np.array_equal(m.known, env.known_map), "reveal mismatch"
    print("PASS grid_mode_matches_env_reveal")


def test_lidar_mode_marks_free_and_obstacles():
    grid = np.zeros((9, 9), dtype=np.int8)
    grid[4, 6] = OBSTACLE
    backend = MockBackend(n_beams=180)
    backend.reset(grid, [(4, 4)], 0.25)
    m = OccupancyMapper(9, 9, 0.25, "lidar", 1)
    rays = backend.raycast(backend.get_poses()[0])
    m.update_lidar_mode(rays, backend.get_poses()[0], (4, 4))
    assert m.known[4, 4] == FREE and m.known[4, 5] == FREE, "near cells free"
    # (4,6) is outside the radius-1 window, so it must stay unknown
    assert m.known[4, 6] == UNKNOWN, "window not respected"
    grid2 = np.zeros((9, 9), dtype=np.int8)
    grid2[4, 5] = OBSTACLE
    backend.reset(grid2, [(4, 4)], 0.25)
    m2 = OccupancyMapper(9, 9, 0.25, "lidar", 1)
    rays = backend.raycast(backend.get_poses()[0])
    m2.update_lidar_mode(rays, backend.get_poses()[0], (4, 4))
    assert m2.known[4, 5] == OBSTACLE, "adjacent obstacle not detected"
    print("PASS lidar_mode_marks_free_and_obstacles")


def test_mock_drive_reaches_targets():
    grid = np.zeros((9, 9), dtype=np.int8)
    b = MockBackend(vmax=0.5)
    b.reset(grid, [(1, 1), (7, 7)], 0.25)
    tgt = np.array([cell_center((1, 2), 0.25), cell_center((7, 6), 0.25)])
    t = b.drive_to(tgt, 0.02, 5.0)
    assert t > 0 and np.allclose(b.get_poses(), tgt), "did not arrive"
    print("PASS mock_drive_reaches_targets")


def test_noise_perturbs_reported_cells():
    """With large sigma the reported cells must sometimes differ."""
    env = RendezvousEnv(EnvConfig(num_robots=3))
    env.reset(seed=7)
    grid, starts = env.grid_map.copy(), [tuple(p) for p in env.positions]

    class _NullModel:
        def predict(self, obs, deterministic=False):
            return np.array([4, 4, 4]), None

    cfg = RunnerConfig(reveal="grid", noise_sigma=0.5, max_steps=1)
    r = LockstepRunner(_NullModel(), MockBackend(), grid, starts, cfg,
                       noise_rng=np.random.default_rng(0))
    seen = set()
    for _ in range(40):
        seen.add(tuple(r._reported_cells()))
    assert len(seen) > 1, "noise had no effect on reported cells"
    print("PASS noise_perturbs_reported_cells")


def test_success_predicate_matches_env():
    cells = [(3, 3), (3, 4), (4, 3), (4, 4)]
    area, (mx, my, side) = bounding_square(cells)
    assert area == 4 and side == 2
    grid = np.zeros((9, 9), dtype=np.int8)
    assert square_free(grid, mx, my, side)
    grid[4, 4] = OBSTACLE
    assert not square_free(grid, mx, my, side)
    print("PASS success_predicate_matches_env")


def test_parity_with_env_paper():
    """THE headline test: identical trajectories under oracle reveal.

    Same map, same starts, same torch seed, mock backend, 'grid' reveal,
    no noise -> the continuous runner must visit exactly the cells the
    plain env visits and agree on success and step count.
    """
    try:
        from stable_baselines3 import PPO
        import torch
    except ImportError:
        print("SKIP parity_with_env_paper (SB3 not installed)")
        return
    model_path = (HERE.parents[0] / "runs_paper" / "final2m" / "N4_M20"
                  / "seed0" / "model.zip")
    if not model_path.exists():
        print("SKIP parity_with_env_paper (final2m N4 model not found)")
        return
    model = PPO.load(str(model_path), device="cpu")

    for ep in (0, 1, 2):
        map_seed = 10_000 + ep
        # --- reference rollout in the training env ---
        env = RendezvousEnv(EnvConfig(num_robots=4, rows=20, cols=20))
        torch.manual_seed((map_seed * 1000) % (2 ** 31))
        obs, _ = env.reset(seed=map_seed)
        grid = env.grid_map.copy()
        starts = [tuple(p) for p in env.positions]
        ref_cells, terminated, truncated, info = [], False, False, {}
        while not (terminated or truncated):
            a, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(a)
            ref_cells.append([tuple(p) for p in env.positions])
        ref_success = bool(info.get("is_success", False))

        # --- continuous rollout through the bridge ---
        torch.manual_seed((map_seed * 1000) % (2 ** 31))
        cfg = RunnerConfig(cell_size=0.25, reveal="grid", noise_sigma=0.0)
        runner = LockstepRunner(model, MockBackend(), grid, starts, cfg)
        # instrument: capture cells after each step
        bridge_cells = []
        orig = runner._scan_all
        steps = 0
        res = None
        # re-implement run() loop minimally to record per-step cells
        while steps < cfg.max_steps:
            steps += 1
            o = {"known_map": runner.mapper.known.copy(),
                 "robot_positions": np.array(runner._reported_cells(),
                                             dtype=np.int32)}
            action, _ = model.predict(o, deterministic=False)
            cur = list(runner.cells)
            targets = []
            for i in range(runner.n):
                dx, dy = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
                          4: (0, 0)}[int(action[i])]
                nx = int(np.clip(cur[i][0] + dx, 0, runner.rows - 1))
                ny = int(np.clip(cur[i][1] + dy, 0, runner.cols - 1))
                if runner.mapper.known[nx, ny] == OBSTACLE:
                    targets.append(cur[i])
                else:
                    targets.append((nx, ny))
            resolve_conflicts(cur, targets)
            pts = np.array([cell_center(t, cfg.cell_size) for t in targets])
            runner.backend.drive_to(pts, cfg.arrive_tol,
                                    cfg.max_sim_s_per_step)
            poses = runner.backend.get_poses()
            runner.cells = [pose_to_cell(poses[i], cfg.cell_size,
                                         runner.rows, runner.cols)
                            for i in range(runner.n)]
            orig()
            bridge_cells.append(list(runner.cells))
            area, (mx, my, side) = bounding_square(runner.cells)
            if area <= cfg.threshold_area and square_free(runner.grid, mx,
                                                          my, side):
                res = True
                break
        bridge_success = bool(res)

        assert len(bridge_cells) == len(ref_cells), (
            f"ep{ep}: step count {len(bridge_cells)} != {len(ref_cells)}")
        assert bridge_cells == ref_cells, f"ep{ep}: trajectories diverge"
        assert bridge_success == ref_success, f"ep{ep}: success differs"
        print(f"   ep{ep}: {len(ref_cells)} steps, success={ref_success}, "
              f"trajectories identical")
    print("PASS parity_with_env_paper")


if __name__ == "__main__":
    test_cell_pose_roundtrip()
    test_conflicts_delegate_to_env()
    test_grid_mode_matches_env_reveal()
    test_lidar_mode_marks_free_and_obstacles()
    test_mock_drive_reaches_targets()
    test_noise_perturbs_reported_cells()
    test_success_predicate_matches_env()
    test_parity_with_env_paper()
    print("\nAll bridge tests passed.")
