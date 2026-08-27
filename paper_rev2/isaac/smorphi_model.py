"""Smorphi robot model for the Isaac captures: the author's CAD mesh plus
procedural add-ons.

``build_smorphi(stage, prim_path, color)`` is the single entry point for
the robot geometry.  It references ref/cad/smorphi_lowpoly.usd (made
from the exported STL by stl_to_usd.py: metres, Z-up, origin at the
footprint centre on the floor, +X forward, 170 x 170 x 204.6 mm) under
one Xform and adds, as UsdGeom primitives, the parts the CAD predates
(see ref/smorphi_platform.jpg, ref/smorphi_details.jpg): the LiDAR puck
on its red mount, the orange genderless docking bracket on each side
face, the front webcam and ArUco plate, and a coloured trim along the
top-plate edges carrying the robot's identity colour.  Everything is
visual only (no colliders), so PhysX raycasts never see a robot.

Plate heights in the mesh (vertex-density peaks): bottom plate top
~0.070, middle ~0.130, top plate top ~0.175.

Self-test (renders close-ups into ../media_isaac_v2/):
    ~/isaac-sim*/python.sh smorphi_model.py
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
MESH_USD = HERE / "ref/cad/smorphi_lowpoly.usd"
BODY_MATERIAL = "/World/Looks/smorphi_body"

TOP_PLATE_Z = 0.175            # top surface of the top plate
BLACK = (0.03, 0.03, 0.035)
WHITE = (0.92, 0.92, 0.90)
MOUNT_RED = (0.78, 0.07, 0.07)
DOCK_ORANGE = (0.96, 0.40, 0.03)


def solid_material(stage, color, roughness=0.75, metallic=0.0):
    """Matte UsdPreviewSurface for a flat colour, cached under /World/Looks."""
    from pxr import Gf, Sdf, UsdShade
    key = "_".join(f"{int(round(c * 255)):02x}" for c in color)
    path = f"/World/Looks/solid_{key}"
    if not stage.GetPrimAtPath(path):
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[float(c) for c in color]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),
                                                  "surface")
    return UsdShade.Material(stage.GetPrimAtPath(path))


def build_smorphi(stage, prim_path: str, color):
    """Create the robot under ``prim_path``; returns the root Xform.

    The caller owns the root's transform ops (translate/orient it there).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    def paint(gprim, c):
        gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*c)]))
        UsdShade.MaterialBindingAPI.Apply(gprim.GetPrim()).Bind(
            solid_material(stage, c))

    def cube(name, pos, size, c):
        g = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        g.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(g.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        paint(g, c)
        return g

    def cyl(name, pos, axis, radius, height, c):
        g = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
        g.CreateAxisAttr(axis)
        g.CreateRadiusAttr(radius)
        g.CreateHeightAttr(height)
        UsdGeom.Xformable(g.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))
        paint(g, c)
        return g

    def ball(name, pos, radius, c):
        g = UsdGeom.Sphere.Define(stage, f"{prim_path}/{name}")
        g.CreateRadiusAttr(radius)
        UsdGeom.Xformable(g.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))
        paint(g, c)
        return g

    root = UsdGeom.Xform.Define(stage, prim_path)

    # --- CAD body: referenced mesh with one uniform material ---------- #
    if not MESH_USD.exists():
        raise FileNotFoundError(f"{MESH_USD} missing; run stl_to_usd.py")
    cad = UsdGeom.Xform.Define(stage, f"{prim_path}/cad")
    cad.GetPrim().GetReferences().AddReference(str(MESH_USD))
    if not stage.GetPrimAtPath(BODY_MATERIAL):
        mat = UsdShade.Material.Define(stage, BODY_MATERIAL)
        sh = UsdShade.Shader.Define(stage, BODY_MATERIAL + "/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.60, 0.61, 0.64))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.25)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.50)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),
                                                  "surface")
    mat = UsdShade.Material(stage.GetPrimAtPath(BODY_MATERIAL))
    UsdShade.MaterialBindingAPI.Apply(cad.GetPrim()).Bind(mat)

    # --- LiDAR puck on red standoffs + mount (puck top at 0.230) ------ #
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        cyl(f"lidar_post_{i}", (0.024 * sx, 0.024 * sy, TOP_PLATE_Z + 0.0095),
            "Z", 0.0045, 0.019, MOUNT_RED)
    cube("lidar_mount", (0, 0, TOP_PLATE_Z + 0.022), (0.060, 0.060, 0.006),
         MOUNT_RED)
    cyl("lidar_puck", (0, 0, TOP_PLATE_Z + 0.040), "Z", 0.035, 0.030, BLACK)

    # --- docking mechanism on each side face, mid height -------------- #
    faces = [(0.087, 0.0), (-0.087, 0.0), (0.0, 0.087), (0.0, -0.087)]
    for i, (fx, fy) in enumerate(faces):
        along_x = fy == 0.0
        cube(f"dock_plate_{i}", (fx, fy, 0.100),
             (0.004, 0.030, 0.060) if along_x else (0.030, 0.004, 0.060),
             DOCK_ORANGE)
        px = fx + (0.013 if fx > 0 else -0.013 if fx < 0 else 0.0)
        py = fy + (0.013 if fy > 0 else -0.013 if fy < 0 else 0.0)
        ball(f"dock_pin_{i}a", (px, py, 0.086), 0.010, DOCK_ORANGE)
        ball(f"dock_pin_{i}b", (px, py, 0.114), 0.010, DOCK_ORANGE)
        for j, hz in enumerate((0.095, 0.105)):
            cyl(f"dock_hole_{i}{j}", (fx, fy, hz), "X" if along_x else "Y",
                0.005, 0.006, (0.02, 0.02, 0.02))

    # --- front (+X) webcam and ArUco plate ---------------------------- #
    cube("cam_body", (0.088, -0.040, 0.076), (0.012, 0.030, 0.010), BLACK)
    cyl("cam_lens", (0.0945, -0.040, 0.076), "X", 0.004, 0.003,
        (0.10, 0.10, 0.12))
    cube("aruco_black", (0.088, -0.036, 0.040), (0.003, 0.056, 0.056), BLACK)
    cube("aruco_white", (0.0895, -0.036, 0.040), (0.002, 0.044, 0.044), WHITE)
    for j, (oy, oz) in enumerate(((-0.011, 0.011), (0.011, 0.011),
                                  (0.011, -0.011), (0.0, -0.011))):
        cube(f"aruco_cell_{j}", (0.0905, -0.036 + oy, 0.040 + oz),
             (0.0015, 0.011, 0.011), BLACK)

    # --- identity trim along the four top-plate edges ----------------- #
    z = TOP_PLATE_Z + 0.0015
    cube("trim_px", (0.082, 0, z), (0.006, 0.170, 0.003), color)
    cube("trim_nx", (-0.082, 0, z), (0.006, 0.170, 0.003), color)
    cube("trim_py", (0, 0.082, z), (0.170, 0.006, 0.003), color)
    cube("trim_ny", (0, -0.082, z), (0.170, 0.006, 0.003), color)
    return root


if __name__ == "__main__":
    # close-up test renders: 3/4 view and top view of a single robot
    import sys

    import numpy as np

    sys.path.insert(0, str(HERE))
    from capture_media import aim_camera, PALETTE  # noqa: E402

    windowed = "--windowed" in sys.argv
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not windowed})

    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom, UsdLux
    from PIL import Image

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = world.stage
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(400.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1500.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set((0.0, -35.0, 30.0))

    build_smorphi(stage, "/World/smorphi", PALETTE[0])
    world.reset()
    for _ in range(3):
        world.step(render=False)

    out = HERE.parents[0] / "media_isaac_v2"
    out.mkdir(exist_ok=True)
    shots = {
        "34": dict(pos=(0.50, -0.40, 0.36), target=(0.0, 0.0, 0.11),
                   up=(0.0, 0.0, 1.0), focal=30.0),
        "top": dict(pos=(0.0, 0.0, 0.90), target=(0.0, 0.0, 0.0),
                    up=(-1.0, 0.0, 0.0), focal=35.0),
    }
    cams = {}
    for name, s in shots.items():
        cam = Camera(prim_path=f"/World/cam_{name}",
                     position=np.array(s["pos"]), resolution=(1280, 960))
        cam.initialize()
        aim_camera(stage, f"/World/cam_{name}", s["pos"], s["target"],
                   s["up"], focal_mm=s["focal"])
        cams[name] = cam
    if windowed:
        try:
            import omni.kit.viewport.utility as vpu
            vpu.get_active_viewport().camera_path = "/World/cam_34"
        except Exception as e:                       # cosmetic only
            print("viewport camera not set:", e)
    for _ in range(40 if not windowed else 240):
        world.step(render=True)
    for name, cam in cams.items():
        arr = np.asarray(cam.get_rgba())
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr[..., :3]).save(
            out / f"smorphi_model_test_{name}.png")
        print("wrote", out / f"smorphi_model_test_{name}.png")
    app.close()
