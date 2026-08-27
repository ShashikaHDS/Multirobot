"""Decimate a binary STL by vertex clustering (pure numpy).

    python decimate_stl.py --inp ref/cad/smorphi_merged.stl \
        --out ref/cad/smorphi_lowpoly.stl --cell 1.0

Vertices are snapped to a cubic grid of --cell (file units, mm here),
coincident vertices merge, degenerate and duplicate triangles are dropped.
At 1 mm on a 170 mm robot the silhouette is preserved for camera
distances of a metre or more while the over-tessellated PCB and plate
surfaces collapse to a few thousand triangles.
"""
import argparse
from pathlib import Path

import numpy as np

STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)),
                      ("attr", "<u2")])


def read_binary_stl(path: Path):
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        return np.frombuffer(f.read(n * 50), dtype=STL_DTYPE)


def write_binary_stl(path: Path, verts, faces):
    tri = np.zeros(len(faces), dtype=STL_DTYPE)
    v = verts[faces]                                    # (F, 3, 3)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(norm > 0, n / np.maximum(norm, 1e-12), 0.0)
    tri["normal"] = n.astype("<f4")
    tri["v"] = v.astype("<f4")
    with open(path, "wb") as f:
        f.write(b"smorphi lowpoly (vertex clustering)".ljust(80, b" "))
        f.write(np.array([len(faces)], dtype="<u4").tobytes())
        f.write(tri.tobytes())


def decimate(tris, cell: float):
    v = tris["v"].reshape(-1, 3).astype(np.float64)
    keys = np.floor(v / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    # representative vertex = mean of the cluster members
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, v)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    verts = sums / counts[:, None]
    faces = inv.reshape(-1, 3)
    # drop degenerate triangles (two or more corners in one cluster)
    ok = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
          & (faces[:, 0] != faces[:, 2]))
    faces = faces[ok]
    # drop duplicate triangles (orientation-insensitive)
    key = np.sort(faces, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    faces = faces[np.sort(first)]
    return verts, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=1.0)
    args = ap.parse_args()
    tris = read_binary_stl(Path(args.inp))
    verts, faces = decimate(tris, args.cell)
    write_binary_stl(Path(args.out), verts, faces)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    print(f"{len(tris):,} -> {len(faces):,} triangles "
          f"({len(faces) / len(tris):.1%}), {len(verts):,} vertices")
    print(f"bbox size {np.round(hi - lo, 2).tolist()}")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
