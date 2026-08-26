"""Headless media capture for the Isaac validation: stills + MP4 clips.

Replays one seeded episode of the standard protocol (same oracle maps,
torch seeding, and lockstep loop as run_validation.py) through a
rendering variant of IsaacBackend, and writes:

  <out>/top.mp4, <out>/persp.mp4        top-down and 3/4-view clips
  <out>/still_<cam>_step<k>.png         stills at --still-steps
  <out>/still_<cam>_final.png           final (rendezvous) frame

Robots get distinct palette colors and drop per-step breadcrumb spheres,
so the final still doubles as a trajectory figure.  bridge.py and
isaac_env.py are untouched; the backend here only overrides scene
cosmetics and the drive loop (render=True + frame grabs).

  ~/isaac-sim*/python.sh capture_media.py --config N4_M20 --train-seed 0 \
      --map 0 --sample 0 --still-steps 1,9,18,27 --out ../media_isaac
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

from env_paper import RendezvousEnv, EnvConfig  # noqa: E402
from bridge import LockstepRunner, RunnerConfig  # noqa: E402
from run_validation import load_policy, EVAL_SEED_BASE  # noqa: E402
from isaac_env import IsaacBackend  # noqa: E402

PALETTE = [
    (0.80, 0.04, 0.04),   # red
    (0.04, 0.25, 0.85),   # blue
    (0.02, 0.45, 0.15),   # green
    (0.90, 0.55, 0.00),   # amber
    (0.45, 0.10, 0.60),   # purple
]


def _look_at_quat(pos, target, up_hint=(0.0, 0.0, 1.0)):
    """wxyz quaternion for camera_axes='world' (+X fwd, +Y left, +Z up)."""
    x = np.asarray(target, float) - np.asarray(pos, float)
    x /= np.linalg.norm(x)
    up = np.asarray(up_hint, float)
    z = up - np.dot(up, x) * x
    n = np.linalg.norm(z)
    if n < 1e-6:                     # forward parallel to up hint
        z = np.array([-1.0, 0.0, 0.0]) - x[0] * x
        n = np.linalg.norm(z)
    z /= n
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)  # columns = camera axes in world
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s,
                         (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = np.empty(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


class CaptureBackend(IsaacBackend):
    """IsaacBackend + colored robots, breadcrumbs, cameras, frame grabs."""

    def __init__(self, out_dir: Path, stride: int = 2, fps: int = 30,
                 breadcrumbs: bool = True, **kw):
        super().__init__(**kw)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stride = stride
        self.fps = fps
        self.breadcrumbs = breadcrumbs
        self._tick = 0
        self._cams = {}
        self._writers = {}
        self._crumb_idx = 0
        self._Sphere = None
        self._policy_step = 0
        self.still_steps = set()

        try:
            from isaacsim.core.api.objects import VisualSphere
        except ImportError:
            from omni.isaac.core.objects import VisualSphere
        self._Sphere = VisualSphere

        from pxr import UsdGeom, UsdLux
        stage = self.world.stage
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(400.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun.CreateIntensityAttr(1500.0)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(
            (0.0, -35.0, 30.0))

    # cosmetics: same geometry as the parent, but colored ------------- #
    def _build_obstacles(self, grid, cell_size):
        from isaac_env import OBSTACLE_HEIGHT
        for prim in self._obstacle_prims:
            self.world.scene.remove_object(prim.name)
        self._obstacle_prims = []
        rows, cols = grid.shape
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    cube = self._FixedCuboid(
                        prim_path=f"/World/obst_{idx}",
                        name=f"obst_{idx}",
                        position=np.array([(r + 0.5) * cell_size,
                                           (c + 0.5) * cell_size,
                                           OBSTACLE_HEIGHT / 2]),
                        scale=np.array([cell_size, cell_size,
                                        OBSTACLE_HEIGHT]),
                        color=np.array([0.28, 0.30, 0.36]),
                    )
                    self.world.scene.add(cube)
                    self._obstacle_prims.append(cube)
                    idx += 1

    def _build_robots(self, n, cell_size):
        for prim in self._robot_prims:
            self.world.scene.remove_object(prim.name)
        self._robot_prims = []
        size = cell_size * 0.6
        for i in range(n):
            cube = self._VisualCuboid(
                prim_path=f"/World/robot_{i}",
                name=f"robot_{i}",
                position=np.array([0.0, 0.0, size / 2]),
                scale=np.array([size, size, size]),
                color=np.array(PALETTE[i % len(PALETTE)]),
            )
            self.world.scene.add(cube)
            self._robot_prims.append(cube)

    # cameras --------------------------------------------------------- #
    def _make_cameras(self, rows, cols):
        try:
            from isaacsim.sensors.camera import Camera
        except ImportError:
            from omni.isaac.sensor import Camera
        cx = rows * self.cell_size / 2.0
        cy = cols * self.cell_size / 2.0
        span = max(rows, cols) * self.cell_size
        # coverage = 2 * dist * (aperture/2) / focal, aperture 20.955mm:
        # top at 1.32*span with f=24 sees ~1.15*span; persp f=19 is wide
        specs = {
            "top": dict(position=np.array([cx, cy, span * 1.32]),
                        target=(cx, cy, 0.0), resolution=(1080, 1080),
                        up_hint=(-1.0, 0.0, 0.0), focal_mm=24.0),
            "persp": dict(position=np.array([cx - span * 1.10, cy,
                                             span * 0.85]),
                          target=(cx, cy, 0.0), resolution=(1280, 720),
                          up_hint=(0.0, 0.0, 1.0), focal_mm=19.0),
        }
        import imageio
        from pxr import UsdGeom
        stage = self.world.stage
        for name, s in specs.items():
            cam = Camera(prim_path=f"/World/cam_{name}",
                         position=s["position"],
                         resolution=s["resolution"])
            cam.initialize()
            cam.set_world_pose(position=s["position"],
                               orientation=_look_at_quat(
                                   s["position"], s["target"],
                                   s["up_hint"]),
                               camera_axes="world")
            UsdGeom.Camera(stage.GetPrimAtPath(
                f"/World/cam_{name}")).GetFocalLengthAttr().Set(
                s["focal_mm"])
            self._cams[name] = cam
            self._writers[name] = imageio.get_writer(
                str(self.out_dir / f"{name}.mp4"), fps=self.fps,
                codec="libx264", quality=8, pixelformat="yuv420p")
        for _ in range(12):                    # renderer warm-up
            self.world.step(render=True)

    def reset(self, grid, starts_cells, cell_size):
        super().reset(grid, starts_cells, cell_size)
        self._apply_poses()           # world.reset() zeroed visual prims
        self._make_border(*self.grid.shape)
        if not self._cams:
            self._make_cameras(*self.grid.shape)
        self.world.step(render=True)

    def _make_border(self, rows, cols):
        """Thin visual-only frame marking the arena extent (no collider,
        so PhysX raycasts are untouched)."""
        if self.world.scene.object_exists("border_0"):
            return
        w = self.cell_size * 0.12
        h = 0.02
        X = rows * self.cell_size
        Y = cols * self.cell_size
        col = np.array([0.15, 0.17, 0.20])
        segs = [
            ((-w / 2, Y / 2), (w, Y + 2 * w)),
            ((X + w / 2, Y / 2), (w, Y + 2 * w)),
            ((X / 2, -w / 2), (X, w)),
            ((X / 2, Y + w / 2), (X, w)),
        ]
        for i, ((px, py), (sx, sy)) in enumerate(segs):
            b = self._VisualCuboid(
                prim_path=f"/World/border_{i}", name=f"border_{i}",
                position=np.array([px, py, h / 2]),
                scale=np.array([sx, sy, h]), color=col)
            self.world.scene.add(b)

    # frame plumbing --------------------------------------------------- #
    def _frame(self, name):
        arr = np.asarray(self._cams[name].get_rgba())
        if arr.size == 0:
            return None
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        return arr[..., :3]

    def _capture_tick(self):
        self._tick += 1
        if self._tick % self.stride:
            return
        for name in self._cams:
            f = self._frame(name)
            if f is not None:
                self._writers[name].append_data(f)

    def _save_stills(self, label):
        from PIL import Image
        for name in self._cams:
            f = self._frame(name)
            if f is not None:
                Image.fromarray(f).save(
                    self.out_dir / f"still_{name}_{label}.png")

    # drive loop: parent kinematics + render/capture/breadcrumbs ------- #
    def drive_to(self, targets_m, tol_m, max_sim_s):
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
            self._apply_poses()
            self.world.step(render=True)
            self._capture_tick()
            t += self.dt
        delta = targets_m - self.poses
        dist = np.linalg.norm(delta, axis=1)
        self.poses[dist <= tol_m] = targets_m[dist <= tol_m]
        self._apply_poses()
        self._policy_step += 1
        if self.breadcrumbs:
            self._drop_breadcrumbs()
        if self._policy_step in self.still_steps:
            self.world.step(render=True)
            self._save_stills(f"step{self._policy_step:03d}")
        return t

    def _drop_breadcrumbs(self):
        r = self.cell_size * 0.11
        for i in range(len(self._robot_prims)):
            col = np.array(PALETTE[i % len(PALETTE)]) * 0.65
            s = self._Sphere(
                prim_path=f"/World/crumb_{self._crumb_idx}",
                name=f"crumb_{self._crumb_idx}",
                position=np.array([self.poses[i][0], self.poses[i][1], r]),
                radius=r, color=col)
            self.world.scene.add(s)
            self._crumb_idx += 1

    def finish(self):
        for _ in range(6):
            self.world.step(render=True)
        self._save_stills("final")
        for w in self._writers.values():
            w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="N4_M20")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--map", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tag", default="final2m")
    ap.add_argument("--logdir", default=str(HERE.parents[0] / "runs_paper"))
    ap.add_argument("--reveal", choices=["lidar", "grid"], default="lidar")
    ap.add_argument("--cell-size", type=float, default=0.25)
    ap.add_argument("--vmax", type=float, default=0.5)
    ap.add_argument("--n-beams", type=int, default=72)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--still-steps", default="1",
                    help="comma-separated policy steps at which to save stills")
    ap.add_argument("--no-breadcrumbs", action="store_true")
    ap.add_argument("--out", default=str(HERE.parents[0] / "media_isaac"))
    args = ap.parse_args()

    n = int(args.config.split("_")[0][1:])
    m = int(args.config.split("_")[1][1:])
    model_path = (Path(args.logdir) / args.tag / args.config
                  / f"seed{args.train_seed}" / "model.zip")
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    backend = CaptureBackend(out_dir=Path(args.out), stride=args.stride,
                             fps=args.fps,
                             breadcrumbs=not args.no_breadcrumbs,
                             vmax=args.vmax, n_beams=args.n_beams,
                             headless=True)
    backend.still_steps = {int(x) for x in args.still_steps.split(",") if x}

    oracle = RendezvousEnv(EnvConfig(num_robots=n, rows=m, cols=m))
    model = load_policy(model_path, oracle)
    map_seed = EVAL_SEED_BASE + args.map
    oracle.reset(seed=map_seed)
    grid = oracle.grid_map.copy()
    starts = [tuple(p) for p in oracle.positions]

    import torch
    torch.manual_seed((map_seed * 1000 + args.sample) % (2 ** 31))
    cfg = RunnerConfig(cell_size=args.cell_size, reveal=args.reveal,
                       noise_sigma=0.0, n_beams=args.n_beams)
    runner = LockstepRunner(
        model, backend, grid, starts, cfg,
        noise_rng=np.random.default_rng(map_seed * 1000 + args.sample))
    backend._save_stills("step000")
    r = runner.run()
    backend.finish()
    print(f"episode: success={r.success} steps={r.steps} "
          f"sim_time={r.sim_time_s:.1f}s")
    print("media in", backend.out_dir)
    backend.close()


if __name__ == "__main__":
    main()
