"""Unit tests for the canonical rendezvous env (run: python test_env_paper.py).

Each test constructs small fixed maps so every collision/reward branch is
exercised deterministically. Written as plain asserts so no pytest install
is required (pytest will also pick them up if available).
"""

import numpy as np

from env_paper import RendezvousEnv, EnvConfig, RewardConfig, UNKNOWN, FREE, OBSTACLE, MapGen

FREE5 = np.zeros((7, 7), dtype=np.int8)          # 7x7 all-free arena


def make_env(starts, grid=None, n=None, max_steps=300, threshold=4):
    starts = np.array(starts, dtype=np.int32)
    n = n or len(starts)
    grid = FREE5 if grid is None else np.array(grid, dtype=np.int8)
    cfg = EnvConfig(num_robots=n, rows=grid.shape[0], cols=grid.shape[1],
                    threshold_area=threshold, max_steps=max_steps)
    env = RendezvousEnv(cfg, fixed_map=grid, fixed_starts=starts)
    env.reset(seed=0)
    return env


def test_determinism():
    cfg = EnvConfig(num_robots=4)
    a = RendezvousEnv(cfg)
    b = RendezvousEnv(cfg)
    oa, _ = a.reset(seed=123)
    ob, _ = b.reset(seed=123)
    assert np.array_equal(a.grid_map, b.grid_map), "same seed -> same map"
    assert np.array_equal(oa["robot_positions"], ob["robot_positions"])
    rng = np.random.default_rng(7)
    for _ in range(50):
        act = rng.integers(0, 5, size=4)
        ra = a.step(act)
        rb = b.step(act)
        assert ra[1] == rb[1] and np.array_equal(
            ra[0]["robot_positions"], rb[0]["robot_positions"])
        if ra[2] or ra[3]:
            break
    print("PASS determinism")


def test_unknown_init():
    env = RendezvousEnv(EnvConfig(num_robots=3))
    obs, _ = env.reset(seed=1)
    km = obs["known_map"]
    assert (km == UNKNOWN).any(), "most of the map should start unknown"
    # every revealed cell must match ground truth; each robot reveals 3x3
    revealed = km != UNKNOWN
    assert (km[revealed] == env.grid_map[revealed]).all()
    for p in obs["robot_positions"]:
        assert km[p[0], p[1]] != UNKNOWN
    print("PASS unknown_init")


def test_obstacle_collision():
    grid = FREE5.copy()
    grid[2, 3] = OBSTACLE
    # robot 0 at (2,2) moves right into the obstacle; robot 1 idles far away
    env = make_env([[2, 2], [6, 6]], grid=grid)
    obs, r, term, trunc, info = env.step([3, 4])          # right, stay
    assert tuple(obs["robot_positions"][0]) == (2, 2), "move reverted"
    assert info["obstacle_collisions"] == 1
    assert r <= RewardConfig().collide_obstacle, f"collision must reach reward, got {r}"
    print("PASS obstacle_collision")


def test_same_target_conflict():
    # robots at (3,2) and (3,4) both move toward (3,3)
    env = make_env([[3, 2], [3, 4], [6, 6]], n=3)
    obs, r, *_ , info = env.step([3, 2, 4])               # right, left, stay
    assert tuple(obs["robot_positions"][0]) == (3, 2)
    assert tuple(obs["robot_positions"][1]) == (3, 4)
    assert info["robot_collisions"] == 2
    print("PASS same_target_conflict")


def test_swap_conflict():
    env = make_env([[3, 2], [3, 3], [6, 6]], n=3)
    obs, r, *_, info = env.step([3, 2, 4])                # 0 right, 1 left = swap
    assert tuple(obs["robot_positions"][0]) == (3, 2)
    assert tuple(obs["robot_positions"][1]) == (3, 3)
    assert info["robot_collisions"] == 2
    print("PASS swap_conflict")


def test_move_into_stationary():
    env = make_env([[3, 2], [3, 3], [6, 6]], n=3)
    obs, r, *_, info = env.step([3, 4, 4])                # 0 right into staying 1
    assert tuple(obs["robot_positions"][0]) == (3, 2)
    assert info["robot_collisions"] == 1, "only the mover collides"
    print("PASS move_into_stationary")


def test_follow_vacated_cell_allowed():
    env = make_env([[3, 2], [3, 3], [6, 6]], n=3)
    obs, r, *_, info = env.step([3, 3, 4])                # both move right in a chain
    assert tuple(obs["robot_positions"][0]) == (3, 3)
    assert tuple(obs["robot_positions"][1]) == (3, 4)
    assert info["robot_collisions"] == 0
    print("PASS follow_vacated_cell_allowed")


def test_revert_cascade():
    # 0 and 1 collide head-on (swap); 2 was moving into the cell 1 vacates.
    # After 1 is reverted, 2's move must also be reverted (cascade).
    env = make_env([[3, 2], [3, 3], [2, 3], [6, 6]], n=4)
    obs, r, *_, info = env.step([3, 2, 1, 4])   # 0 right, 1 left (swap), 2 down into (3,3)
    assert tuple(obs["robot_positions"][2]) == (2, 3), "cascade revert"
    assert info["robot_collisions"] == 3
    print("PASS revert_cascade")


def test_reward_accounting():
    rw = RewardConfig()
    # shrink: 2 robots far apart, one steps closer -> area decreases below best
    env = make_env([[1, 1], [1, 5], [5, 1]], n=3, threshold=1)
    _, r, *_ = env.step([4, 2, 4])            # robot 1 left: extent 4->3, area 25->25? x-extent 5
    # area before: side max(4,4)+1=5 ->25 ; after: max(4,3)+1=5 -> still 25 (x extent governs)
    assert r == 0.0, f"no improvement, no growth -> 0, got {r}"
    _, r, *_ = env.step([4, 4, 0])            # robot 2 up: x-extent 4->3, side=max(3,3)+1=4 area16<25
    assert r == rw.area_decrease, f"shrink -> +20, got {r}"
    _, r, *_ = env.step([4, 4, 1])            # robot 2 back down: area grows 16->25
    assert r == rw.area_increase, f"growth -> -0.5, got {r}"
    # collision + shrink must ACCUMULATE in the same step (the legacy env
    # overwrote the -5 with the area term -- the exact bug we fixed)
    grid = FREE5.copy()
    grid[1, 2] = OBSTACLE
    env2 = make_env([[1, 1], [1, 5], [5, 1]], grid=grid, n=3, threshold=1)
    # robot0 hits the obstacle (-5) while robots 1+2 shrink both extents:
    # side 5->4, area 25->16 < best (+20) => net +15
    _, r, *_, info = env2.step([3, 2, 0])
    assert info["obstacle_collisions"] == 1
    expected = rw.collide_obstacle + rw.area_decrease
    assert r == expected, f"collision+shrink must accumulate: want {expected}, got {r}"
    # collision alone with no area change -> exactly -5
    _, r, *_, info = env2.step([3, 4, 4])
    assert info["obstacle_collisions"] == 1
    assert r == rw.collide_obstacle, f"collision penalty must survive, got {r}"
    print("PASS reward_accounting")


def test_goal_and_termination():
    rw = RewardConfig()
    # 3 robots nearly together; start area 9 (>4), one move reaches area 4
    env = make_env([[2, 2], [2, 3], [4, 3]], n=3, threshold=4)
    obs, r, term, trunc, info = env.step([4, 4, 4])       # all stay: area 9, no goal
    assert not term and r == 0.0
    obs, r, term, trunc, info = env.step([4, 4, 0])       # robot2 (4,3)->(3,3): side 2, area 4
    assert term and info["is_success"]
    assert r == rw.goal, f"goal reward exactly +100, got {r}"
    print("PASS goal_and_termination")


def test_goal_blocked_by_obstacle():
    grid = FREE5.copy()
    grid[2, 2] = OBSTACLE                                  # inside the bounding square
    env = make_env([[1, 1], [1, 3], [3, 1]], grid=grid, n=3, threshold=9)
    obs, r, term, trunc, info = env.step([4, 4, 4])        # area 9 <= 9 but square not free
    assert not term, "square containing an obstacle is not a valid goal"
    print("PASS goal_blocked_by_obstacle")


def test_truncation():
    env = make_env([[1, 1], [5, 5]], max_steps=5, threshold=1)
    for i in range(5):
        obs, r, term, trunc, info = env.step([4, 4])
        assert not term
    assert trunc, "episode must truncate at max_steps"
    print("PASS truncation")


def test_spawn_not_solved():
    cfg = EnvConfig(num_robots=4, threshold_area=16)
    env = RendezvousEnv(cfg)
    for s in range(30):
        _, info = env.reset(seed=s)
        assert info["bounding_area"] > 16, "episodes must not start solved"
    print("PASS spawn_not_solved")


def test_mapgen_connectivity():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = MapGen.generate(20, 20, 5, (2, 10), 3.0, rng)
        free = np.argwhere(g == FREE)
        assert len(free) > 0
        # BFS from first free cell must reach all free cells
        seen = np.zeros_like(g, dtype=bool)
        stack = [tuple(free[0])]
        seen[tuple(free[0])] = True
        while stack:
            x, y = stack.pop()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 20 and 0 <= ny < 20 and not seen[nx, ny] and g[nx, ny] == FREE:
                    seen[nx, ny] = True
                    stack.append((nx, ny))
        assert seen.sum() == len(free), "free space must be one component"
        assert g[0, :].max() == FREE and g[-1, :].max() == FREE, "border ring free"
    print("PASS mapgen_connectivity")


def test_sb3_env_checker():
    try:
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("SKIP sb3_env_checker (SB3 not installed)")
        return
    env = RendezvousEnv(EnvConfig(num_robots=4), seed=0)
    check_env(env, warn=True)
    print("PASS sb3_env_checker")


if __name__ == "__main__":
    test_determinism()
    test_unknown_init()
    test_obstacle_collision()
    test_same_target_conflict()
    test_swap_conflict()
    test_move_into_stationary()
    test_follow_vacated_cell_allowed()
    test_revert_cascade()
    test_reward_accounting()
    test_goal_and_termination()
    test_goal_blocked_by_obstacle()
    test_truncation()
    test_spawn_not_solved()
    test_mapgen_connectivity()
    test_sb3_env_checker()
    print("\nAll tests passed.")
