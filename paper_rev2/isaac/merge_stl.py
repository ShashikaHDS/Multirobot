"""Merge the per-component STL files exported from the Smorphi SolidWorks
assembly into one binary STL for Isaac Sim, dropping fasteners.

    python merge_stl.py [--cad ref/cad] [--out ref/cad/smorphi_merged.stl]

Pure numpy binary-STL reader/writer (no extra dependencies).  Prints the
triangle count and bounding box so the units and the 170 x 170 x 230 mm
footprint can be checked, and writes <out>.json with that metadata.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DROP = re.compile(r"Cap Screw|Wing Screw|Shaft Sleeve", re.I)

STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)),
                      ("attr", "<u2")])


def read_binary_stl(path: Path):
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        data = np.frombuffer(f.read(n * 50), dtype=STL_DTYPE)
    if len(data) != n:
        raise ValueError(f"{path.name}: expected {n} triangles, got {len(data)}")
    return data


def write_binary_stl(path: Path, tris):
    header = b"smorphi merged (fasteners dropped)".ljust(80, b" ")
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.array([len(tris)], dtype="<u4").tobytes())
        f.write(tris.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cad", default=str(HERE / "ref" / "cad"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="Tiley_Single - ")
    args = ap.parse_args()
    cad = Path(args.cad)
    out = Path(args.out) if args.out else cad / "smorphi_merged.stl"

    parts, dropped = [], []
    for p in sorted(cad.glob(f"{args.prefix}*.STL")):
        if DROP.search(p.name):
            dropped.append(p.name)
            continue
        parts.append(p)
    if not parts:
        raise SystemExit(f"no component STLs under {cad}")

    chunks, per_part = [], []
    for p in parts:
        t = read_binary_stl(p)
        chunks.append(t)
        per_part.append((p.name[len(args.prefix):], len(t)))
    tris = np.concatenate(chunks)
    verts = tris["v"].reshape(-1, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    size = hi - lo

    write_binary_stl(out, tris)
    meta = {
        "triangles": int(len(tris)),
        "parts_kept": len(parts),
        "parts_dropped": len(dropped),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "size": size.tolist(),
        "largest_parts": sorted(per_part, key=lambda x: -x[1])[:8],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"kept {len(parts)} parts, dropped {len(dropped)} fasteners, "
          f"{len(tris):,} triangles")
    print(f"bbox min {np.round(lo, 2).tolist()}  max {np.round(hi, 2).tolist()}")
    print(f"size {np.round(size, 2).tolist()}  (file units, mm expected)")
    print("largest parts:")
    for name, n in meta["largest_parts"]:
        print(f"  {n:>9,}  {name}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
