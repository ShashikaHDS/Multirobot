"""Media capture for the Isaac validation: stills + MP4 clips, v2.

Replays one seeded episode of the standard protocol (same oracle maps,
torch seeding, and lockstep loop as run_validation.py) through a
rendering variant of IsaacBackend, and writes:

  <out>/top.mp4, <out>/persp.mp4        top-down and 3/4-view clips
  <out>/still_<cam>_step<k>.png         stills at --still-steps (+ step 0)
  <out>/still_<cam>_final.png           final (rendezvous) frame

Scene dressing, all visual-only prims (no colliders, so the PhysX
raycasts that drive the policy are untouched):
  * robots are the Smorphi model from smorphi_model.build_smorphi (CAD
    mesh + add-ons), rotated to face their last move direction, with a
    breadcrumb trail in the robot's identity colour
  * the current step's LiDAR returns of each robot: hit spheres and thin
    beams in the robot's colour (previous step's are replaced)
  * an explored-area overlay: translucent pale-yellow floor tiles on
    every cell the fleet's OccupancyMapper knows as free, added as the
    map is revealed (known obstacles are the grey blocks already)

bridge.py and isaac_env.py are untouched; the backend here overrides
scene cosmetics and the drive loop (render=True + frame grabs), and
CaptureRunner only mirrors the mapper's known mask into the scene.

  ~/isaac-sim*/python.sh capture_media.py --config N4_M20 --train-seed 0 \
      --map 3 --sample 0 --still-steps 5,10,15,20 --out ../media_isaac_v2
  add --windowed to watch it live in the Isaac Sim window while recording
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

from env_paper import RendezvousEnv, EnvConfig, FREE  # noqa: E402
from bridge import LockstepRunner, RunnerConfig  # noqa: E402
from run_validation import load_policy, EVAL_SEED_BASE  # noqa: E402
from isaac_env import IsaacBackend, OBSTACLE_HEIGHT, RAY_Z  # noqa: E402
from smorphi_model import build_smorphi, solid_material  # noqa: E402

PALETTE = [
    (0.80, 0.04, 0.04),   # red
    (0.04, 0.25, 0.85),   # blue
    (0.02, 0.45, 0.15),   # green
    (0.90, 0.55, 0.00),   # amber
    (0.45, 0.10, 0.60),   # purple
]
TILE_COLOR = (0.98, 0.88, 0.40)
TILE_OPACITY = 0.35
HIT_OPACITY = 0.55
BEAM_OPACITY = 0.30


def aim_camera(stage, prim_path, pos, target, up, focal_mm=None,
               near=0.02, far=500.0):
    """Point and configure a camera prim (convention-proof).

    Builds the view matrix with Gf.Matrix4d.SetLookAt (USD camera:
    -Z forward, +Y image-up) and writes its inverse as the prim's single
    transform op, replacing whatever ops the wrapper installed.  Also
    sets the focal length and the clipping range: USD's default near
    plane is 1.0 stage unit (1 m), which blanks out close-up shots.
    """
    from pxr import Gf, UsdGeom
    view = Gf.Matrix4d()
    view.SetLookAt(Gf.Vec3d(*[float(v) for v in pos]),
                   Gf.Vec3d(*[float(v) for v in target]),
                   Gf.Vec3d(*[float(v) for v in up]))
    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    cam = UsdGeom.Camera(prim)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(float(near), float(far)))
    if focal_mm is not None:
        cam.GetFocalLengthAttr().Set(float(focal_mm))


class CaptureBackend(IsaacBackend):
    """IsaacBackend + Smorphi robots, LiDAR/explored overlays, cameras."""

    def __init__(self, out_dir: Path, stride: int = 2, fps: int = 30,
                 quality: int = 6, breadcrumbs: bool = True,
                 top_res=(1080, 1080), persp_res=(1920, 1080), **kw):
        super().__init__(**kw)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stride = stride
        self.fps = fps
        self.quality = quality
        self.breadcrumbs = breadcrumbs
        self.top_res = tuple(top_res)
        self.persp_res = tuple(persp_res)
        self.headless = kw.get("headless", True)
        self._tick = 0
        self._cams = {}
        self._writers = {}
        self._crumb_idx = 0
        self._policy_step = 0
        self.still_steps = set()
        self._robot_ops = []          # (translate op, rotateZ op) per robot
        self._headings = np.zeros((0, 2))
        self._hits = {}               # robot -> list of hit-sphere translate ops
        self._beams = {}              # robot -> BasisCurves
        self._tiles = set()

        from pxr import UsdGeom, UsdLux
        stage = self.world.stage
        UsdLux.DomeLight.Define(stage, "/World/DomeLight") \
            .CreateIntensityAttr(400.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun.CreateIntensityAttr(1500.0)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(
            (0.0, -35.0, 30.0))

    # ---------------- raw-prim helpers ---------------- #
    def _prim_exists(self, path):
        return bool(self.world.stage.GetPrimAtPath(path))

    def _raw_cube(self, path, pos, size, color, opacity=None):
        from pxr import Gf, UsdGeom, Vt
        g = UsdGeom.Cube.Define(self.world.stage, path)
        g.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(g.GetPrim())
        t = xf.AddTranslateOp()
        t.Set(Gf.Vec3d(*[float(v) for v in pos]))
        xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        g.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        if opacity is not None:
            g.GetDisplayOpacityAttr().Set(Vt.FloatArray([float(opacity)]))
        return t

    def _raw_sphere(self, path, pos, radius, color, opacity=None):
        from pxr import Gf, UsdGeom, Vt
        g = UsdGeom.Sphere.Define(self.world.stage, path)
        g.CreateRadiusAttr(float(radius))
        t = UsdGeom.Xformable(g.GetPrim()).AddTranslateOp()
        t.Set(Gf.Vec3d(*[float(v) for v in pos]))
        g.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        if opacity is not None:
            g.GetDisplayOpacityAttr().Set(Vt.FloatArray([float(opacity)]))
        return t

    # ---------------- scene: obstacles, robots, border ---------------- #
    def _build_obstacles(self, grid, cell_size):
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
        from pxr import UsdGeom
        stage = self.world.stage
        for i in range(len(self._robot_prims)):
            stage.RemovePrim(f"/World/robot_{i}")
        self._robot_prims = []
        self._robot_ops = []
        for i in range(n):
            root = build_smorphi(stage, f"/World/robot_{i}",
                                 PALETTE[i % len(PALETTE)])
            xf = UsdGeom.Xformable(root.GetPrim())
            self._robot_ops.append((xf.AddTranslateOp(), xf.AddRotateZOp()))
            self._robot_prims.append(root.GetPrim())
        self._headings = np.tile([1.0, 0.0], (n, 1))

    def _apply_poses(self):
        from pxr import Gf
        for i, (t, r) in enumerate(self._robot_ops):
            t.Set(Gf.Vec3d(float(self.poses[i][0]), float(self.poses[i][1]),
                           0.0))
            r.Set(float(math.degrees(math.atan2(self._headings[i][1],
                                                self._headings[i][0]))))

    def _make_border(self, rows, cols):
        if self._prim_exists("/World/border_0"):
            return
        w = self.cell_size * 0.12
        h = 0.02
        X = rows * self.cell_size
        Y = cols * self.cell_size
        col = (0.15, 0.17, 0.20)
        segs = [((-w / 2, Y / 2), (w, Y + 2 * w)),
                ((X + w / 2, Y / 2), (w, Y + 2 * w)),
                ((X / 2, -w / 2), (X, w)),
                ((X / 2, Y + w / 2), (X, w))]
        for i, ((px, py), (sx, sy)) in enumerate(segs):
            self._raw_cube(f"/World/border_{i}", (px, py, h / 2),
                           (sx, sy, h), col)

    def reset(self, grid, starts_cells, cell_size):
        super().reset(grid, starts_cells, cell_size)
        self._apply_poses()
        self._make_border(*self.grid.shape)
        if not self._cams:
            self._make_cameras(*self.grid.shape)
        self.world.step(render=True)

    # ---------------- cameras ---------------- #
    def _make_cameras(self, rows, cols):
        try:
            from isaacsim.sensors.camera import Camera
        except ImportError:
            from omni.isaac.sensor import Camera
        import imageio
        cx = rows * self.cell_size / 2.0
        cy = cols * self.cell_size / 2.0
        span = max(rows, cols) * self.cell_size
        # coverage = 2 * dist * (aperture/2) / focal, aperture 20.955mm:
        # top at 1.32*span with f=24 sees ~1.15*span; persp f=19 is wide
        specs = {
            "top": dict(pos=(cx, cy, span * 1.32), target=(cx, cy, 0.0),
                        up=(-1.0, 0.0, 0.0), focal=24.0, res=self.top_res),
            "persp": dict(pos=(cx - span * 1.10, cy, span * 0.85),
                          target=(cx, cy, 0.0), up=(0.0, 0.0, 1.0),
                          focal=19.0, res=self.persp_res),
        }
        stage = self.world.stage
        for name, s in specs.items():
            path = f"/World/cam_{name}"
            cam = Camera(prim_path=path, position=np.array(s["pos"]),
                         resolution=s["res"])
            cam.initialize()
            aim_camera(stage, path, s["pos"], s["target"], s["up"],
                       focal_mm=s["focal"])
            self._cams[name] = cam
            self._writers[name] = imageio.get_writer(
                str(self.out_dir / f"{name}.mp4"), fps=self.fps,
                codec="libx264", quality=self.quality, pixelformat="yuv420p")
        if not self.headless:                # let the author watch the 3/4 cam
            try:
                import omni.kit.viewport.utility as vpu
                vpu.get_active_viewport().camera_path = "/World/cam_persp"
            except Exception as e:
                print("viewport camera not set:", e)
        for _ in range(12):                  # renderer warm-up
            self.world.step(render=True)

    # ---------------- LiDAR overlay ---------------- #
    def raycast(self, origin_m):
        out = super().raycast(origin_m)
        if len(self.poses):
            i = int(np.argmin(np.linalg.norm(self.poses - origin_m[:2],
                                             axis=1)))
            self._show_lidar(i, origin_m, out)
        return out

    def _ensure_lidar_prims(self, i):
        if i in self._hits:
            return
        from pxr import Gf, UsdGeom, Vt
        stage = self.world.stage
        color = PALETTE[i % len(PALETTE)]
        UsdGeom.Xform.Define(stage, f"/World/lidar_{i}")
        r = self.cell_size * 0.06
        self._hits[i] = [
            self._raw_sphere(f"/World/lidar_{i}/hit_{b}", (0, 0, -5.0), r,
                             color, HIT_OPACITY)
            for b in range(self.n_beams)]
        curves = UsdGeom.BasisCurves.Define(stage, f"/World/lidar_{i}/beams")
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateCurveVertexCountsAttr(Vt.IntArray([2] * self.n_beams))
        curves.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, -5.0)]
                                              * (2 * self.n_beams)))
        curves.CreateWidthsAttr(Vt.FloatArray([0.004]))
        curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curves.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        curves.GetDisplayOpacityAttr().Set(Vt.FloatArray([BEAM_OPACITY]))
        self._beams[i] = curves

    def _show_lidar(self, i, origin_m, rays):
        from pxr import Gf, Vt
        self._ensure_lidar_prims(i)
        ox, oy = float(origin_m[0]), float(origin_m[1])
        pts = []
        for b, (hit, x, y) in enumerate(rays):
            if hit:
                self._hits[i][b].Set(Gf.Vec3d(float(x), float(y), RAY_Z))
                pts += [Gf.Vec3f(ox, oy, RAY_Z), Gf.Vec3f(float(x), float(y),
                                                          RAY_Z)]
            else:
                self._hits[i][b].Set(Gf.Vec3d(0.0, 0.0, -5.0))
                pts += [Gf.Vec3f(ox, oy, RAY_Z), Gf.Vec3f(ox, oy, RAY_Z)]
        self._beams[i].GetPointsAttr().Set(Vt.Vec3fArray(pts))

    # ---------------- explored-area overlay ---------------- #
    def update_known(self, known):
        cs = self.cell_size
        for r, c in zip(*np.nonzero(known == FREE)):
            if (r, c) in self._tiles:
                continue
            self._tiles.add((r, c))
            self._raw_cube(f"/World/tiles/t_{r}_{c}",
                           ((r + 0.5) * cs, (c + 0.5) * cs, 0.002),
                           (cs, cs, 0.004), TILE_COLOR, TILE_OPACITY)

    # ---------------- frame plumbing ---------------- #
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
        for _ in range(3):        # newly defined prims need extra passes
            self.world.step(render=True)
        for name in self._cams:
            f = self._frame(name)
            if f is not None:
                Image.fromarray(f).save(
                    self.out_dir / f"still_{name}_{label}.png")

    # ---------------- drive loop ---------------- #
    def drive_to(self, targets_m, tol_m, max_sim_s):
        t = 0.0
        targets_m = np.asarray(targets_m, dtype=np.float64)
        delta0 = targets_m - self.poses
        for i in range(len(self.poses)):
            d = np.linalg.norm(delta0[i])
            if d > 1e-6:
                self._headings[i] = delta0[i] / d
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
            self._save_stills(f"step{self._policy_step:03d}")
        return t

    def _drop_breadcrumbs(self):
        r = self.cell_size * 0.11
        for i in range(len(self._robot_ops)):
            col = tuple(np.array(PALETTE[i % len(PALETTE)]) * 0.65)
            self._raw_sphere(f"/World/crumbs/c_{self._crumb_idx}",
                             (self.poses[i][0], self.poses[i][1], r), r, col)
            self._crumb_idx += 1

    def finish(self):
        for _ in range(6):
            self.world.step(render=True)
        self._save_stills("final")
        for w in self._writers.values():
            w.close()


class CaptureRunner(LockstepRunner):
    """LockstepRunner that mirrors the mapper's known mask into the scene."""

    def _scan_all(self):
        super()._scan_all()
        self.backend.update_known(self.mapper.known)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="N4_M20")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--map", type=int, default=3)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tag", default="final2m")
    ap.add_argument("--logdir", default=str(HERE.parents[0] / "runs_paper"))
    ap.add_argument("--reveal", choices=["lidar", "grid"], default="lidar")
    ap.add_argument("--cell-size", type=float, default=0.25)
    ap.add_argument("--vmax", type=float, default=0.5)
    ap.add_argument("--n-beams", type=int, default=72)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=6,
                    help="imageio/ffmpeg quality 0-10")
    ap.add_argument("--still-steps", default="5,10,15,20",
                    help="comma-separated policy steps at which to save stills")
    ap.add_argument("--no-breadcrumbs", action="store_true")
    ap.add_argument("--windowed", action="store_true",
                    help="show the Isaac Sim window while recording")
    ap.add_argument("--out", default=str(HERE.parents[0] / "media_isaac_v2"))
    args = ap.parse_args()

    n = int(args.config.split("_")[0][1:])
    m = int(args.config.split("_")[1][1:])
    model_path = (Path(args.logdir) / args.tag / args.config
                  / f"seed{args.train_seed}" / "model.zip")
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    backend = CaptureBackend(out_dir=Path(args.out), stride=args.stride,
                             fps=args.fps, quality=args.quality,
                             breadcrumbs=not args.no_breadcrumbs,
                             vmax=args.vmax, n_beams=args.n_beams,
                             headless=not args.windowed)
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
    runner = CaptureRunner(
        model, backend, grid, starts, cfg,
        noise_rng=np.random.default_rng(map_seed * 1000 + args.sample))
    backend._save_stills("step000")
    r = runner.run()
    backend.finish()
    print(f"episode: success={r.success} steps={r.steps} "
          f"sim_time={r.sim_time_s:.1f}s", flush=True)
    print("media in", backend.out_dir, flush=True)
    backend.close()


if __name__ == "__main__":
    main()
