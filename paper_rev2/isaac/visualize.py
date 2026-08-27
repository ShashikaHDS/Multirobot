"""Run one rendezvous episode in Isaac Sim for a chosen fleet size, watch it
live, and optionally record video.

    python.sh visualize.py --n 4                   # N=4 on a 20x20 map, windowed, records MP4s
    python.sh visualize.py --n 5 --m 25 --map 3    # N=5 on a 25x25 map, held-out map 3
    python.sh visualize.py --n 3 --no-video        # watch only, no recording
    python.sh visualize.py --n 4 --headless        # record without opening the window

Fleet sizes with trained models: N = 2, 3, 4, 5 on 20x20 and N = 5 on
25x25 (the paper's final2m set).  --map selects the held-out map
(generator seed 10000 + map), --sample the seeded stochastic rollout, so
every run is exactly reproducible.  Recording uses Isaac Sim's own frame
capture (top-down and perspective MP4s plus stills) via capture_media.py.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

VALID = {(2, 20), (3, 20), (4, 20), (5, 20), (5, 25)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="number of robots")
    ap.add_argument("--m", type=int, default=None,
                    help="map size (default 20, or 25 for the N=5 25x25 model)")
    ap.add_argument("--map", type=int, default=0, help="held-out map index 0..19")
    ap.add_argument("--sample", type=int, default=0, help="seeded rollout index")
    ap.add_argument("--train-seed", type=int, default=0, help="training seed 0..2")
    ap.add_argument("--no-video", action="store_true", help="watch only, no MP4s")
    ap.add_argument("--headless", action="store_true",
                    help="do not open the Isaac window (recording still works)")
    ap.add_argument("--stills", default="0,5,10,15,20",
                    help="policy steps at which to save still frames")
    ap.add_argument("--out", default=None,
                    help="output folder (default media_isaac_N<n>_M<m>_map<map>)")
    args = ap.parse_args()

    m = args.m if args.m is not None else 20
    if (args.n, m) not in VALID:
        sys.exit(f"no trained model for N={args.n} on {m}x{m}; valid: "
                 + ", ".join(f"N={n} {mm}x{mm}" for n, mm in sorted(VALID)))
    out = args.out or str(HERE.parents[0]
                          / f"media_isaac_N{args.n}_M{m}_map{args.map}")

    argv = ["capture_media.py",
            "--config", f"N{args.n}_M{m}",
            "--train-seed", str(args.train_seed),
            "--map", str(args.map),
            "--sample", str(args.sample),
            "--still-steps", args.stills,
            "--out", out]
    if not args.headless:
        argv.append("--windowed")
    if args.no_video:
        argv.append("--no-video")

    sys.path.insert(0, str(HERE))
    import capture_media
    sys.argv = argv
    print(f"N={args.n} on {m}x{m}, map {args.map}, sample {args.sample}, "
          f"{'headless' if args.headless else 'windowed'}, "
          f"{'no video' if args.no_video else 'recording'} -> {out}")
    capture_media.main()


if __name__ == "__main__":
    main()
