#!/usr/bin/env python3
"""One-file launcher for the Isaac Sim rendezvous demos: GUI and shell.

Runs under the SYSTEM python (needs only tkinter, no extra packages) and
spawns Isaac Sim's own python (python.sh) for the actual work, so it can
also install the packages that python needs (stable-baselines3,
gymnasium, imageio, imageio-ffmpeg).

    python3 isaac_launcher.py                 # open the control panel
    python3 isaac_launcher.py --n 4           # shell: watch N=4 live, record MP4s
    python3 isaac_launcher.py --n 5 --m 25 --map 0 --no-video
    python3 isaac_launcher.py --n 4 --headless --stills 0,10,20
    python3 isaac_launcher.py --n 4 --starts "0,0;0,19;19,0;19,19"   # your own start cells
    python3 isaac_launcher.py --n 4 --grid-file ../maps/mymap.json      # a map drawn in the GUI
    python3 isaac_launcher.py --install-deps  # just install the Isaac-python packages
    python3 isaac_launcher.py --validate smoke|full|noise
    python3 isaac_launcher.py --model-test    # Smorphi close-up render

Isaac Sim is located automatically (ISAAC_PYTHON env var, ~/isaacsim,
~/isaac-sim*, ~/.local/share/ov/pkg/isaac_sim-*, /isaac-sim, /opt/isaacsim)
or picked in the GUI.  Everything else (visualize.py, capture_media.py,
run_validation.py, smorphi_model.py) lives next to this file.
"""
from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent

PIP_PACKAGES = ["stable-baselines3", "gymnasium", "imageio", "imageio-ffmpeg"]
IMPORT_NAMES = ["stable_baselines3", "gymnasium", "imageio", "imageio_ffmpeg"]

VALIDATION = {
    "smoke": ("2 maps, N=4, 2 samples (~1 min)",
              [["--maps", "2", "--n", "4", "--m", "20", "--samples", "2",
                "--out", "results_isaac_smoke2"]]),
    "full": ("10 maps x N3/N4/N5 x 5 samples (~1 h)",
             [["--maps", "10", "--n", "3,4,5", "--m", "20,25",
               "--samples", "5", "--out", "results_isaac"]]),
    "noise": ("sigma 0.00/0.05/0.10 m on N4_M20 (~35 min)",
              [["--maps", "10", "--n", "4", "--m", "20", "--samples", "5",
                "--noise", s, "--out", f"results_isaac_n{s}"]
               for s in ("0.00", "0.05", "0.10")]),
}

# log lines hidden unless "verbose" is on (kit extension chatter)
NOISE = re.compile(r"^\[\d+(\.\d+)?s\] \[ext: |\[Warning\]|^Warning: running in conda"
                   r"|^If conda is desired")
PY_STDERR = re.compile(r"^.*\[Error\] \[omni\.kit\.app\._impl\] \[py stderr\]: ?")


# --------------------------------------------------------------------- #
# Isaac python discovery and dependency handling                        #
# --------------------------------------------------------------------- #
def find_isaac_python() -> str | None:
    env = os.environ.get("ISAAC_PYTHON")
    if env and Path(env).exists():
        return env
    home = Path.home()
    patterns = [
        str(home / "isaacsim/python.sh"),
        str(home / ".local/share/ov/pkg/isaac_sim-*/python.sh"),
        str(home / "isaac-sim*/python.sh"),
        "/isaac-sim/python.sh",
        "/opt/isaacsim/python.sh",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat), reverse=True)     # newest version first
        for h in hits:
            if os.access(h, os.X_OK):
                return h
    return None


def missing_deps(isaac_py: str) -> list[str]:
    """Import each package inside Isaac's python; return the missing ones."""
    code = ("import importlib.util,sys\n"
            "bad=[m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]\n"
            "print('MISSING:'+','.join(bad))")
    try:
        out = subprocess.run([isaac_py, "-c", code] + IMPORT_NAMES,
                             capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [f"(could not run {isaac_py}: {e})"]
    for line in out.stdout.splitlines():
        if line.startswith("MISSING:"):
            return [m for m in line[8:].split(",") if m]
    return ["(dependency check failed: " + (out.stderr.strip().splitlines() or ["?"])[-1] + ")"]


def model_configs() -> dict[str, list[int]]:
    """{'N4_M20': [0, 1, 2], ...} from runs_paper/final2m; falls back to the paper set."""
    found = {}
    for model in sorted((PAPER / "runs_paper/final2m").glob("N*_M*/seed*/model.zip")):
        cfg = model.parents[1].name
        found.setdefault(cfg, []).append(int(model.parent.name[4:]))
    if not found:
        found = {c: [0, 1, 2] for c in ("N2_M20", "N3_M20", "N4_M20", "N5_M20", "N5_M25")}
    return {k: sorted(v) for k, v in found.items()}


# --------------------------------------------------------------------- #
# command construction                                                  #
# --------------------------------------------------------------------- #
def cmd_visualize(isaac_py, n, m, map_i, sample, train_seed, stills, video,
                  windowed, quality, out, starts=None, grid_file=None):
    cmd = [isaac_py, str(HERE / "capture_media.py"),
           "--config", f"N{n}_M{m}", "--train-seed", str(train_seed),
           "--map", str(map_i), "--sample", str(sample),
           "--still-steps", stills, "--quality", str(quality),
           "--out", out or str(PAPER / f"media_isaac_N{n}_M{m}_map{map_i}")]
    if grid_file:
        cmd += ["--grid-file", grid_file]
    if starts:
        cmd += ["--starts", starts]
    if windowed:
        cmd.append("--windowed")
    if not video:
        cmd.append("--no-video")
    return cmd


def cmds_validate(isaac_py, mode):
    return [[isaac_py, str(HERE / "run_validation.py"), "--backend", "isaac",
             "--reveal", "lidar"] + args for args in VALIDATION[mode][1]]


def cmd_model_test(isaac_py, windowed):
    return [isaac_py, str(HERE / "smorphi_model.py")] + (["--windowed"] if windowed else [])


def cmd_install(isaac_py):
    return [isaac_py, "-m", "pip", "install"] + PIP_PACKAGES


PALETTE_HEX = ["#cc0a0a", "#0a40d9", "#05732b", "#e68c00", "#731a99"]


def load_map(isaac_py: str, n: int, m: int, map_i: int):
    """Grid + protocol start cells for (n, m, map) from env_paper, run in
    Isaac's python (system python lacks gymnasium).  Returns dict or str error."""
    code = ("import sys, json\n"
            f"sys.path.insert(0, {str(PAPER)!r})\n"
            "from env_paper import RendezvousEnv, EnvConfig\n"
            f"env = RendezvousEnv(EnvConfig(num_robots={n}, rows={m}, cols={m}))\n"
            f"env.reset(seed={10000 + map_i})\n"
            "print('MAP:' + json.dumps({'grid': env.grid_map.tolist(), "
            "'starts': env.positions.tolist()}))\n")
    try:
        out = subprocess.run([isaac_py, "-c", code], capture_output=True, text=True,
                             timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"could not run {isaac_py}: {e}"
    for line in out.stdout.splitlines():
        if line.startswith("MAP:"):
            import json
            return json.loads(line[4:])
    tail = (out.stderr.strip().splitlines() or out.stdout.strip().splitlines() or ["?"])[-1]
    return "map load failed: " + tail


def parse_starts(text: str):
    cells = []
    for tok in [t for t in text.replace(" ", "").split(";") if t]:
        r, c = (int(v) for v in tok.split(","))
        cells.append((r, c))
    return cells


def starts_error(cells, grid, n) -> str | None:
    if len(cells) != n:
        return f"{len(cells)} start cells for {n} robots"
    rows, cols = len(grid), len(grid[0])
    for r, c in cells:
        if not (0 <= r < rows and 0 <= c < cols):
            return f"cell {r},{c} is outside the grid"
        if grid[r][c] != 0:
            return f"cell {r},{c} is an obstacle"
    if len(set(cells)) != len(cells):
        return "two robots on the same cell"
    return None


def starts_text(cells) -> str:
    return ";".join(f"{r},{c}" for r, c in cells)


def run_env(windowed: bool) -> dict:
    env = dict(os.environ)
    if windowed and not env.get("DISPLAY"):
        env["DISPLAY"] = ":1"          # the desk monitor when started over ssh
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


# --------------------------------------------------------------------- #
# subprocess runner (streams lines, sequential command queue)           #
# --------------------------------------------------------------------- #
class Runner:
    def __init__(self, on_line, on_done):
        self.on_line = on_line
        self.on_done = on_done
        self.proc = None
        self.pgid = None
        self._stop = False
        self.thread = None

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, commands: list[list[str]], env: dict, cwd: Path):
        self._stop = False
        self.thread = threading.Thread(target=self._run, args=(commands, env, cwd),
                                       daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the current command and everything it spawned.

        python.sh is a bash wrapper that runs the real Isaac process as a
        child, so the whole process group is signalled (SIGTERM, then
        SIGKILL after 8 s), not just the wrapper.
        """
        self._stop = True
        pgid = self.pgid
        if pgid is None:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)                 # still anything alive?
            except ProcessLookupError:
                return
            time.sleep(0.2)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _run(self, commands, env, cwd):
        code = 0
        t0 = time.time()
        for cmd in commands:
            if self._stop:
                code = -1
                break
            self.on_line("$ " + " ".join(shlex.quote(c) for c in cmd))
            try:
                self.proc = subprocess.Popen(cmd, cwd=str(cwd), env=env,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             text=True, bufsize=1,
                                             errors="replace",
                                             start_new_session=True)
            except OSError as e:
                self.on_line(f"!! cannot start: {e}")
                code = 127
                break
            self.pgid = self.proc.pid             # session leader == group id
            for line in self.proc.stdout:
                self.on_line(line.rstrip("\n"))
            code = self.proc.wait()
            try:                                   # no stragglers from the group
                os.killpg(self.pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc = None
            self.pgid = None
            if code != 0:
                self.on_line(f"!! exited with code {code}"
                             + (" (stopped)" if self._stop else ""))
                break
        self.on_done(code, time.time() - t0)


def clean_line(line: str, verbose: bool) -> str | None:
    if not verbose and NOISE.search(line):
        return None
    return PY_STDERR.sub("", line)


# --------------------------------------------------------------------- #
# GUI                                                                   #
# --------------------------------------------------------------------- #
def launch_gui(isaac_py: str | None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    configs = model_configs()
    ns = sorted({int(c.split("_")[0][1:]) for c in configs})

    root = tk.Tk()
    root.title("Isaac Sim rendezvous launcher")
    root.minsize(860, 640)
    lines = queue.Queue()

    # ---- top: Isaac python + dependencies ---------------------------- #
    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="Isaac Sim python:").grid(row=0, column=0, sticky="w")
    py_var = tk.StringVar(value=isaac_py or "")
    ttk.Entry(top, textvariable=py_var, width=70).grid(row=0, column=1, sticky="ew", padx=4)

    def browse():
        p = filedialog.askopenfilename(title="Select Isaac Sim python.sh")
        if p:
            py_var.set(p)
    ttk.Button(top, text="Browse", command=browse).grid(row=0, column=2)
    dep_var = tk.StringVar(value="dependencies: not checked")
    ttk.Label(top, textvariable=dep_var).grid(row=1, column=1, sticky="w", padx=4)
    top.columnconfigure(1, weight=1)

    # ---- tabs --------------------------------------------------------- #
    nb = ttk.Notebook(root)
    nb.pack(fill="x", padx=8)

    # Watch / Record
    w = ttk.Frame(nb, padding=8)
    nb.add(w, text="Watch / Record")
    n_var = tk.StringVar(value="4" if 4 in ns else str(ns[0]))
    m_var = tk.StringVar(value="20")
    map_var = tk.StringVar(value="0")
    sample_var = tk.StringVar(value="0")
    seed_var = tk.StringVar(value="0")
    stills_var = tk.StringVar(value="0,5,10,15,20")
    quality_var = tk.StringVar(value="7")
    video_var = tk.BooleanVar(value=True)
    window_var = tk.BooleanVar(value=True)
    out_var = tk.StringVar(value="")
    out_edited = tk.BooleanVar(value=False)

    def ms_for(n):
        return sorted({int(c.split("_")[1][1:]) for c in configs
                       if int(c.split("_")[0][1:]) == n})

    def refresh_choices(*_):
        n = int(n_var.get())
        ms = ms_for(n)
        m_box["values"] = [str(x) for x in ms]
        if int(m_var.get()) not in ms:
            m_var.set(str(ms[0]))
        seeds = configs.get(f"N{n}_M{m_var.get()}", [0])
        seed_box["values"] = [str(s) for s in seeds]
        if int(seed_var.get()) not in seeds:
            seed_var.set(str(seeds[0]))
        if not out_edited.get():
            out_var.set(str(PAPER / f"media_isaac_N{n}_M{m_var.get()}_map{map_var.get()}"))

    r = 0
    ttk.Label(w, text="Robots (N)").grid(row=r, column=0, sticky="w")
    n_box = ttk.Combobox(w, textvariable=n_var, values=[str(x) for x in ns], width=5,
                         state="readonly")
    n_box.grid(row=r, column=1, sticky="w")
    ttk.Label(w, text="Grid (M)").grid(row=r, column=2, sticky="w", padx=(16, 0))
    m_box = ttk.Combobox(w, textvariable=m_var, width=5, state="readonly")
    m_box.grid(row=r, column=3, sticky="w")
    ttk.Label(w, text="Training seed").grid(row=r, column=4, sticky="w", padx=(16, 0))
    seed_box = ttk.Combobox(w, textvariable=seed_var, width=5, state="readonly")
    seed_box.grid(row=r, column=5, sticky="w")
    r += 1
    ttk.Label(w, text="Held-out map (0-9)").grid(row=r, column=0, sticky="w")
    ttk.Spinbox(w, from_=0, to=9, textvariable=map_var, width=5).grid(row=r, column=1, sticky="w")
    ttk.Label(w, text="Sample").grid(row=r, column=2, sticky="w", padx=(16, 0))
    ttk.Spinbox(w, from_=0, to=99, textvariable=sample_var, width=5).grid(row=r, column=3, sticky="w")
    ttk.Label(w, text="Video quality (1-10)").grid(row=r, column=4, sticky="w", padx=(16, 0))
    ttk.Spinbox(w, from_=1, to=10, textvariable=quality_var, width=5).grid(row=r, column=5, sticky="w")
    r += 1
    ttk.Label(w, text="Stills at steps").grid(row=r, column=0, sticky="w")
    ttk.Entry(w, textvariable=stills_var, width=24).grid(row=r, column=1, columnspan=3, sticky="w")
    ttk.Checkbutton(w, text="Record MP4s", variable=video_var).grid(row=r, column=4, sticky="w", padx=(16, 0))
    ttk.Checkbutton(w, text="Show Isaac window", variable=window_var).grid(row=r, column=5, sticky="w")
    r += 1
    ttk.Label(w, text="Output folder").grid(row=r, column=0, sticky="w")
    out_entry = ttk.Entry(w, textvariable=out_var, width=56)
    out_entry.grid(row=r, column=1, columnspan=4, sticky="ew")
    out_entry.bind("<Key>", lambda e: out_edited.set(True))
    ask_out_var = tk.BooleanVar(value=True)

    def browse_out():
        d = choose_folder("Save videos and stills in ...", out_var.get() or str(PAPER))
        if d:
            out_var.set(d)
            out_edited.set(True)
            append(f"output folder: {d}")
        return d
    outbtns = ttk.Frame(w)
    outbtns.grid(row=r, column=5, sticky="w")
    ttk.Button(outbtns, text="Browse...", command=browse_out).pack(side="left", padx=(6, 0))
    ttk.Checkbutton(outbtns, text="ask on Run", variable=ask_out_var).pack(side="left", padx=6)
    r += 1
    btns = ttk.Frame(w)
    btns.grid(row=r, column=0, columnspan=6, sticky="w", pady=(8, 0))
    run_btn = ttk.Button(btns, text="Run episode")
    run_btn.pack(side="left")
    ttk.Button(btns, text="Open output folder",
               command=lambda: open_folder(out_var.get())).pack(side="left", padx=6)
    ttk.Label(w, text="Watching: the window plays ~2x real time; MP4s are real time. "
                      "A run takes about 40 s.").grid(row=r + 1, column=0, columnspan=6,
                                                     sticky="w", pady=(6, 0))
    # ---- map and start positions: clickable canvas ------------------- #
    sp = ttk.LabelFrame(w, text="Map and start positions", padding=6)
    sp.grid(row=r + 2, column=0, columnspan=6, sticky="ew", pady=(8, 0))
    CANVAS = 320
    canvas = tk.Canvas(sp, width=CANVAS, height=CANVAS, bg="#ffffff",
                       highlightthickness=1, highlightbackground="#888")
    canvas.grid(row=0, column=0, rowspan=12, sticky="nw")
    custom_var = tk.BooleanVar(value=False)
    starts_var = tk.StringVar(value="")
    map_status = tk.StringVar(value="map: not loaded")
    mode_var = tk.StringVar(value="robots")            # robots | draw | erase
    mp = {"key": None, "grid": None, "default": [], "starts": [], "next": 0,
          "loading": None, "modified": False, "rects": []}

    def cell_px():
        return CANVAS / len(mp["grid"])

    def draw_robots():
        canvas.delete("robot")
        if mp["grid"] is None:
            return
        px = cell_px()
        for i, (rr, cc) in enumerate(mp["starts"]):
            col = PALETTE_HEX[i % len(PALETTE_HEX)]
            pad = px * 0.12
            canvas.create_oval(cc * px + pad, rr * px + pad, (cc + 1) * px - pad,
                               (rr + 1) * px - pad, fill=col, tags="robot",
                               outline="#000" if i == mp["next"] else col, width=2)
            canvas.create_text((cc + 0.5) * px, (rr + 0.5) * px, text=str(i + 1),
                               fill="white", tags="robot",
                               font=("sans", max(7, int(px * 0.45)), "bold"))
        starts_var.set(starts_text(mp["starts"]))

    def draw():
        canvas.delete("all")
        grid = mp["grid"]
        if grid is None:
            canvas.create_text(CANVAS / 2, CANVAS / 2, fill="#555",
                               text="map not loaded\n(Load protocol map / New blank map)")
            return
        m = len(grid)
        px = CANVAS / m
        mp["rects"] = [[canvas.create_rectangle(
            cc * px, rr * px, (cc + 1) * px, (rr + 1) * px,
            fill="#1b1f27" if grid[rr][cc] else "#ffffff", outline="#cfd5dd")
            for cc in range(m)] for rr in range(m)]
        draw_robots()

    def set_starts(cells, custom):
        mp["starts"] = list(cells)
        mp["next"] = 0
        custom_var.set(custom)
        draw_robots()

    def set_map(grid, key, default_starts, modified, status):
        mp.update(key=key, grid=[list(row) for row in grid], modified=modified,
                  default=list(default_starts))
        map_status.set(status)
        draw()
        set_starts(default_starts, modified)

    def cell_at(ev):
        if mp["grid"] is None:
            return None
        m = len(mp["grid"])
        px = CANVAS / m
        rr, cc = int(ev.y // px), int(ev.x // px)
        return (rr, cc) if 0 <= rr < m and 0 <= cc < m else None

    def paint(cell, value):
        rr, cc = cell
        if mp["grid"][rr][cc] == value or (value and (rr, cc) in mp["starts"]):
            return
        mp["grid"][rr][cc] = value
        canvas.itemconfig(mp["rects"][rr][cc], fill="#1b1f27" if value else "#ffffff")
        if not mp["modified"]:
            mp["modified"] = True
            map_status.set("map: custom (edited) - will be saved as map.json in the output folder")

    def place_robot(cell):
        if mp["grid"][cell[0]][cell[1]]:
            return
        n = int(n_var.get())
        starts = list(mp["starts"])[:n]
        if cell in starts:                      # click a robot: select it
            mp["next"] = starts.index(cell)
            draw_robots()
            return
        if len(starts) < n:
            starts.append(cell)
            mp["next"] = len(starts) % n
        else:
            starts[mp["next"]] = cell
            mp["next"] = (mp["next"] + 1) % n
        mp["starts"] = starts
        custom_var.set(True)
        draw_robots()

    def on_press(ev, button=1):
        cell = cell_at(ev)
        if cell is None:
            return
        mode = mode_var.get()
        if button == 3 or mode == "erase":
            paint(cell, 0)
        elif mode == "draw":
            paint(cell, 1)
        else:
            place_robot(cell)

    def on_drag(ev, button=1):
        cell = cell_at(ev)
        if cell is None:
            return
        mode = mode_var.get()
        if button == 3 or mode == "erase":
            paint(cell, 0)
        elif mode == "draw":
            paint(cell, 1)
    canvas.bind("<Button-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<Button-3>", lambda e: on_press(e, 3))
    canvas.bind("<B3-Motion>", lambda e: on_drag(e, 3))

    def do_random():
        if mp["grid"] is None:
            return
        import random
        free = [(rr, cc) for rr in range(len(mp["grid"]))
                for cc in range(len(mp["grid"][0])) if mp["grid"][rr][cc] == 0]
        n = int(n_var.get())
        if len(free) < n:
            messagebox.showerror("Start positions", "not enough free cells")
            return
        set_starts(random.sample(free, n), True)

    def request_map(*_):
        p = py_var.get().strip()
        try:
            key = (int(n_var.get()), int(m_var.get()), int(map_var.get()))
        except ValueError:
            return
        if key == mp["key"] or key == mp["loading"]:
            return
        if not p or not Path(p).exists():
            map_status.set("map: set the Isaac python path to load the protocol map")
            return
        mp["loading"] = key
        map_status.set(f"map: loading N{key[0]}_M{key[1]} map {key[2]} ...")

        def work():
            lines.put(("__map__", key, load_map(p, *key)))
        threading.Thread(target=work, daemon=True).start()

    def new_blank():
        m = int(m_var.get())
        set_map([[0] * m for _ in range(m)], ("blank", m), [], True,
                f"map: blank {m}x{m} - draw obstacles (black), then place robots")

    def open_map():
        f = filedialog.askopenfilename(title="Open map", initialdir=str(PAPER / "maps"),
                                       filetypes=[("map json", "*.json")])
        if not f:
            return
        import json
        try:
            data = json.load(open(f))
            grid = data["grid"] if isinstance(data, dict) else data
            starts = [tuple(c) for c in data.get("starts", [])] if isinstance(data, dict) else []
            m = len(grid)
            assert all(len(row) == m for row in grid)
        except Exception as e:
            messagebox.showerror("Open map", f"not a map file: {e}")
            return
        if str(m) in m_box["values"]:
            m_var.set(str(m))
        set_map(grid, ("file", f), starts, True, f"map: {Path(f).name} ({m}x{m}), custom")

    def save_map():
        if mp["grid"] is None:
            return
        (PAPER / "maps").mkdir(exist_ok=True)
        f = filedialog.asksaveasfilename(title="Save map as", initialdir=str(PAPER / "maps"),
                                         defaultextension=".json",
                                         filetypes=[("map json", "*.json")])
        if not f:
            return
        import json
        json.dump({"m": len(mp["grid"]), "grid": mp["grid"],
                   "starts": [list(c) for c in mp["starts"]]}, open(f, "w"))
        map_status.set(f"map: saved {Path(f).name}")

    col = 1
    ttk.Button(sp, text="Load protocol map", command=lambda: (mp.update(key=None, modified=False),
                                                              request_map())
               ).grid(row=0, column=col, sticky="w", padx=8)
    ttk.Button(sp, text="New blank map", command=new_blank).grid(row=1, column=col, sticky="w", padx=8)
    ttk.Button(sp, text="Open map file...", command=open_map).grid(row=2, column=col, sticky="w", padx=8)
    ttk.Button(sp, text="Save map as...", command=save_map).grid(row=3, column=col, sticky="w", padx=8)
    modes = ttk.Frame(sp)
    modes.grid(row=4, column=col, sticky="w", padx=8, pady=(6, 0))
    ttk.Label(modes, text="Click does:").pack(side="left")
    for text, val in (("place robots", "robots"), ("draw obstacles", "draw"), ("erase", "erase")):
        ttk.Radiobutton(modes, text=text, value=val, variable=mode_var).pack(side="left", padx=4)
    ttk.Button(sp, text="Default positions",
               command=lambda: set_starts(mp["default"], False)).grid(row=5, column=col, sticky="w", padx=8)
    ttk.Button(sp, text="Random positions", command=do_random).grid(row=6, column=col, sticky="w", padx=8)
    ttk.Checkbutton(sp, text="Use these start positions (else the protocol's seeded starts)",
                    variable=custom_var).grid(row=7, column=col, sticky="w", padx=8)
    cells_row = ttk.Frame(sp)
    cells_row.grid(row=8, column=col, sticky="w", padx=8)
    ttk.Label(cells_row, text="cells r,c:").pack(side="left")
    ttk.Entry(cells_row, textvariable=starts_var, width=40).pack(side="left", padx=4)
    ttk.Label(sp, textvariable=map_status, wraplength=480).grid(row=9, column=col, sticky="w", padx=8)
    ttk.Label(sp, wraplength=480, text=(
        "White = free, black = obstacle. Drag to draw, right-drag erases. In 'place robots' "
        "mode a click puts the highlighted robot on a free cell (then the next one); clicking "
        "a robot selects it. Rows go down, columns go right, as in the top-view video. An "
        "edited or blank map is saved as map.json next to the run's output.")
              ).grid(row=10, column=col, sticky="w", padx=8, pady=(4, 0))
    draw()

    for v in (n_var, m_var, map_var):
        v.trace_add("write", refresh_choices)
        v.trace_add("write", lambda *_: root.after(400, request_map))
    py_var.trace_add("write", lambda *_: root.after(400, request_map))
    refresh_choices()

    # Validation
    v = ttk.Frame(nb, padding=8)
    nb.add(v, text="Validation (headless)")
    val_btns = {}
    for i, (mode, (desc, _)) in enumerate(VALIDATION.items()):
        val_btns[mode] = ttk.Button(v, text=f"Run {mode}")
        val_btns[mode].grid(row=i, column=0, sticky="w", pady=2)
        ttk.Label(v, text=desc + f"   ->  paper_rev2/results_isaac{'' if mode == 'full' else '_' + ('smoke2' if mode == 'smoke' else 'n<sigma>')}/").grid(row=i, column=1, sticky="w", padx=8)
    ttk.Button(v, text="Open results folder",
               command=lambda: open_folder(str(PAPER))).grid(row=3, column=0, sticky="w", pady=(8, 0))

    # Model / setup
    s = ttk.Frame(nb, padding=8)
    nb.add(s, text="Model & setup")
    model_btn = ttk.Button(s, text="Render Smorphi close-up")
    model_btn.grid(row=0, column=0, sticky="w", pady=2)
    ttk.Label(s, text="writes media_isaac_v2/smorphi_model_test_34.png and _top.png"
              ).grid(row=0, column=1, sticky="w", padx=8)
    usd_btn = ttk.Button(s, text="Rebuild mesh USD from STL")
    usd_btn.grid(row=1, column=0, sticky="w", pady=2)
    ttk.Label(s, text="only after ref/cad/smorphi_lowpoly.stl changes").grid(row=1, column=1, sticky="w", padx=8)
    check_btn = ttk.Button(s, text="Check dependencies")
    check_btn.grid(row=2, column=0, sticky="w", pady=2)
    install_btn = ttk.Button(s, text="Install dependencies")
    install_btn.grid(row=3, column=0, sticky="w", pady=2)
    ttk.Label(s, text="pip install " + " ".join(PIP_PACKAGES) + " into Isaac's python"
              ).grid(row=3, column=1, sticky="w", padx=8)
    test_btn = ttk.Button(s, text="Run bridge tests (plain python)")
    test_btn.grid(row=4, column=0, sticky="w", pady=2)

    # ---- log ---------------------------------------------------------- #
    bar = ttk.Frame(root, padding=(8, 4))
    bar.pack(fill="x")
    status_var = tk.StringVar(value="idle")
    ttk.Label(bar, textvariable=status_var).pack(side="left")
    verbose_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(bar, text="verbose log", variable=verbose_var).pack(side="right")
    stop_btn = ttk.Button(bar, text="Stop", state="disabled")
    stop_btn.pack(side="right", padx=6)
    ttk.Button(bar, text="Clear log", command=lambda: log.delete("1.0", "end")).pack(side="right")
    log = scrolledtext.ScrolledText(root, height=22, font=("monospace", 9))
    log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    all_btns = [run_btn, model_btn, usd_btn, check_btn, install_btn, test_btn] + list(val_btns.values())

    def set_busy(busy):
        for b in all_btns:
            b["state"] = "disabled" if busy else "normal"
        stop_btn["state"] = "normal" if busy else "disabled"

    def append(line):
        log.insert("end", line + "\n")
        log.see("end")

    def on_line(line):
        lines.put(line)

    def on_done(code, secs):
        lines.put(("__done__", code, secs))

    runner = Runner(on_line, on_done)

    def isaac():
        p = py_var.get().strip()
        if not p or not Path(p).exists():
            messagebox.showerror("Isaac Sim python", "Set the path to Isaac Sim's python.sh first.")
            return None
        return p

    def start(commands, windowed=False, label="running"):
        if runner.busy:
            return
        set_busy(True)
        status_var.set(label + " ...")
        runner.start(commands, run_env(windowed), HERE)

    def do_run():
        p = isaac()
        if not p:
            return
        try:
            int(map_var.get()); int(sample_var.get()); int(quality_var.get())
        except ValueError:
            messagebox.showerror("Input", "map, sample and quality must be integers")
            return
        n, m = int(n_var.get()), int(m_var.get())
        if ask_out_var.get() and not browse_out():
            status_var.set("run cancelled (no output folder chosen)")
            return
        out = out_var.get().strip() or str(PAPER / f"media_isaac_N{n}_M{m}_map{map_var.get()}")
        grid_file = None
        starts = None
        custom_map = mp["grid"] is not None and mp["modified"]
        if custom_map:
            if len(mp["grid"]) != m:
                messagebox.showerror("Map", f"the drawn map is {len(mp['grid'])}x{len(mp['grid'])} "
                                            f"but M={m}; pick the matching M or start a new blank map")
                return
            if len(mp["starts"]) != n:
                messagebox.showerror("Start positions",
                                     f"place all {n} robots on the custom map first")
                return
        if custom_var.get() or custom_map:
            key = (n, m, int(map_var.get()))
            if mp["grid"] is None or (not custom_map and mp["key"] != key):
                messagebox.showerror("Start positions", "Load the map for this N / M / map first.")
                return
            try:
                cells = parse_starts(starts_var.get())
            except ValueError:
                messagebox.showerror("Start positions", 'cells must look like "r,c;r,c;..."')
                return
            err = starts_error(cells, mp["grid"], n)
            if err:
                messagebox.showerror("Start positions", err)
                return
            starts = starts_text(cells)
        if custom_map:
            import json
            Path(out).mkdir(parents=True, exist_ok=True)
            grid_file = str(Path(out) / "map.json")
            json.dump({"m": m, "grid": mp["grid"], "starts": [list(c) for c in mp["starts"]]},
                      open(grid_file, "w"))
        cmd = cmd_visualize(p, n, m, map_var.get(), sample_var.get(),
                            seed_var.get(), stills_var.get().replace(" ", ""),
                            video_var.get(), window_var.get(), quality_var.get(),
                            out, starts, grid_file)
        start([cmd], windowed=window_var.get(),
              label=f"N={n} M={m} " + ("custom map" if custom_map else f"map {map_var.get()}")
                    + (" custom starts" if starts else ""))

    def do_validate(mode):
        p = isaac()
        if p:
            start(cmds_validate(p, mode), label=f"validation {mode}")

    def do_check():
        p = isaac()
        if not p:
            return
        dep_var.set("dependencies: checking ...")

        def work():
            miss = missing_deps(p)
            lines.put(("__deps__", miss))
        threading.Thread(target=work, daemon=True).start()

    def do_install():
        p = isaac()
        if p:
            start([cmd_install(p)], label="installing dependencies")

    def do_model():
        p = isaac()
        if p:
            start([cmd_model_test(p, window_var.get())], windowed=window_var.get(),
                  label="model close-up")

    def do_usd():
        p = isaac()
        if p:
            start([[p, str(HERE / "stl_to_usd.py")]], label="rebuilding USD")

    def do_tests():
        start([[sys.executable, str(HERE / "test_bridge.py")]], label="bridge tests")

    run_btn["command"] = do_run
    for mode, b in val_btns.items():
        b["command"] = lambda m=mode: do_validate(m)
    model_btn["command"] = do_model
    usd_btn["command"] = do_usd
    check_btn["command"] = do_check
    install_btn["command"] = do_install
    test_btn["command"] = do_tests
    stop_btn["command"] = lambda: threading.Thread(target=runner.stop, daemon=True).start()

    # log lines, run completion and dependency results all arrive via the queue
    def poll():
        try:
            while True:
                item = lines.get_nowait()
                if isinstance(item, tuple) and item[0] == "__map__":
                    _, key, data = item
                    mp["loading"] = None
                    if isinstance(data, str):
                        map_status.set("map: " + data)
                    elif mp["modified"] and mp["key"] is not None:
                        pass            # user is editing a custom map; keep it
                    else:
                        set_map(data["grid"], key, [tuple(c) for c in data["starts"]], False,
                                f"map: N{key[0]}_M{key[1]} map {key[2]} loaded "
                                f"(seed {10000 + key[2]}); protocol start cells shown")
                    continue
                if isinstance(item, tuple) and item[0] == "__deps__":
                    miss = item[1]
                    dep_var.set("dependencies: all present" if not miss
                                else "dependencies missing: " + ", ".join(miss)
                                + "  (Model & setup -> Install dependencies)")
                elif isinstance(item, tuple):
                    _, code, secs = item
                    status_var.set(f"finished (exit {code}) in {secs:.0f} s" if code == 0
                                   else f"stopped / failed (exit {code}) after {secs:.0f} s")
                    set_busy(False)
                else:
                    shown = clean_line(item, verbose_var.get())
                    if shown is not None:
                        append(shown)
        except queue.Empty:
            pass
        root.after(100, poll)

    append(f"launcher folder: {HERE}")
    if isaac_py:
        append(f"Isaac Sim python: {isaac_py}")
        do_check()
    else:
        append("Isaac Sim not found automatically - set the python.sh path above.")
    def on_close():
        if runner.busy:
            if not messagebox.askyesno("Quit", "A run is still active. Stop it and quit?"):
                return
            runner.stop()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    if os.environ.get("ISAAC_LAUNCHER_SELFTEST"):
        root.after(500, root.destroy)
    poll()
    root.mainloop()


def choose_folder(title: str, initial: str) -> str:
    """Native folder chooser (GNOME's via zenity, KDE's via kdialog); falls
    back to Tk's own dialog, whose Linux version only returns a folder that
    is explicitly selected."""
    import shutil
    initial = str(Path(initial))
    start = initial if Path(initial).is_dir() else str(Path(initial).parent)
    if shutil.which("zenity"):
        r = subprocess.run(["zenity", "--file-selection", "--directory",
                            f"--title={title}", f"--filename={start}/"],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    if shutil.which("kdialog"):
        r = subprocess.run(["kdialog", "--getexistingdirectory", start,
                            "--title", title], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    from tkinter import filedialog
    return filedialog.askdirectory(title=title, initialdir=start, mustexist=False) or ""


def open_folder(path: str):
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except OSError:
        print("open:", path)


# --------------------------------------------------------------------- #
# shell mode                                                            #
# --------------------------------------------------------------------- #
def run_cli(commands, windowed=False) -> int:
    done = {}

    def on_done(code, secs):
        done["code"] = code
        print(f"-- finished (exit {code}) in {secs:.0f} s")

    r = Runner(lambda line: print(line, flush=True) if clean_line(line, False) is not None else None,
               on_done)
    r.start(commands, run_env(windowed), HERE)
    try:
        while r.busy:
            time.sleep(0.2)
    except KeyboardInterrupt:
        r.stop()
        return 130
    return done.get("code", 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, help="robots; runs one episode from the shell (no GUI)")
    ap.add_argument("--m", type=int, default=None, help="grid size (default 20)")
    ap.add_argument("--map", type=int, default=0, help="held-out map 0-9")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--stills", default="0,5,10,15,20")
    ap.add_argument("--quality", type=int, default=7)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--starts", default=None,
                    help='start cells "r,c;r,c;..." (one per robot) instead of the seeded ones')
    ap.add_argument("--grid-file", default=None,
                    help="custom MxM map json (as saved by the GUI) instead of the seeded map")
    ap.add_argument("--install-deps", action="store_true")
    ap.add_argument("--validate", choices=list(VALIDATION))
    ap.add_argument("--model-test", action="store_true")
    ap.add_argument("--isaac-python", default=None, help="path to python.sh")
    args = ap.parse_args()

    isaac_py = args.isaac_python or find_isaac_python()
    shell_mode = any([args.n is not None, args.install_deps, args.validate, args.model_test])
    if not shell_mode:
        launch_gui(isaac_py)
        return 0
    if not isaac_py:
        sys.exit("Isaac Sim python.sh not found; pass --isaac-python or set ISAAC_PYTHON")
    print("Isaac Sim python:", isaac_py)

    if args.install_deps:
        return run_cli([cmd_install(isaac_py)])
    miss = missing_deps(isaac_py)
    if any(m.startswith("(") for m in miss):
        sys.exit("dependency check could not run: " + "; ".join(miss))
    if miss:
        print("installing missing packages:", ", ".join(miss))
        code = run_cli([cmd_install(isaac_py)])
        if code:
            return code
    if args.validate:
        return run_cli(cmds_validate(isaac_py, args.validate))
    if args.model_test:
        return run_cli([cmd_model_test(isaac_py, not args.headless)], windowed=not args.headless)

    m = args.m or 20
    configs = model_configs()
    if f"N{args.n}_M{m}" not in configs:
        sys.exit(f"no trained model N{args.n}_M{m}; available: " + ", ".join(sorted(configs)))
    cmd = cmd_visualize(isaac_py, args.n, m, args.map, args.sample, args.train_seed,
                        args.stills, not args.no_video, not args.headless, args.quality,
                        args.out, args.starts, args.grid_file)
    return run_cli([cmd], windowed=not args.headless)


if __name__ == "__main__":
    sys.exit(main())
