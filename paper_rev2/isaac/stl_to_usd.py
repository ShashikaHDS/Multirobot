"""One-off: convert ref/cad/smorphi_lowpoly.stl (mm, Y-up, SolidWorks) to
ref/cad/smorphi_lowpoly.usd in the sim frame (metres, Z-up, origin at
the footprint centre on the floor).  Pure pxr, no kit needed.

    ~/isaac-sim*/python.sh stl_to_usd.py

Boots a headless kit app only to get pxr; no scene is built.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

try:                                   # pxr only exists inside a kit app
    from pxr import Gf, Usd, UsdGeom, Vt
except ModuleNotFoundError:
    from isaacsim import SimulationApp
    _app = SimulationApp({"headless": True})
    from pxr import Gf, Usd, UsdGeom, Vt

HERE = Path(__file__).resolve().parent
SRC = HERE / "ref/cad/smorphi_lowpoly.stl"
DST = HERE / "ref/cad/smorphi_lowpoly.usd"

# footprint centre (x, z) and floor (y) in the mm Y-up source frame
CENTER_MM = np.array([85.85, 11.97, 85.85])


def read_binary_stl(path: Path):
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tri = np.frombuffer(raw[84:84 + 50 * n], dtype=dt)
    return tri["v"].reshape(-1, 3).astype(np.float64)


def to_sim_frame(v_mm: np.ndarray) -> np.ndarray:
    p = v_mm - CENTER_MM
    p = np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)   # +90 deg about X
    return p * 0.001


def main():
    v = to_sim_frame(read_binary_stl(SRC))
    lo, hi = v.min(0), v.max(0)
    print(f"sim bbox min {lo.round(4)} max {hi.round(4)}")
    assert abs(lo[2]) < 1e-4 and abs(hi[2] - 0.2046) < 5e-4, "frame check"

    # weld coincident vertices (STL repeats every corner)
    key = np.round(v, 6)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    idx = inv.reshape(-1)
    ntri = len(idx) // 3
    tri = idx.reshape(-1, 3)
    # per-face normals from the transformed geometry (flat shading suits CAD)
    p0, p1, p2 = uniq[tri[:, 0]], uniq[tri[:, 1]], uniq[tri[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 1e-12, nrm / np.maximum(ln, 1e-12), [0.0, 0.0, 1.0])

    if DST.exists():
        DST.unlink()
    stage = Usd.Stage.CreateNew(str(DST))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/smorphi")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/smorphi/body")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(uniq.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * ntri))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx.astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in lo]),
                                         Gf.Vec3f(*[float(x) for x in hi])]))
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.78, 0.79, 0.81)]))
    stage.GetRootLayer().Save()
    print(f"wrote {DST} ({DST.stat().st_size / 1e6:.1f} MB): "
          f"{len(uniq)} verts, {ntri} tris")


if __name__ == "__main__":
    main()
