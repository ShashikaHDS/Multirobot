# stc_pygame.py
import sys, time, math, heapq
from collections import deque
import pygame

# ===========================
# STC PLANNER (same file)
# ===========================
FOUR_NEIGHBORS = [(1,0),(-1,0),(0,1),(0,-1)]

def in_bounds(r, c, H, W):
    return 0 <= r < H and 0 <= c < W

def free_cell(grid, r, c, obstacle_values):
    return grid[r][c] not in obstacle_values

def astar(grid, start, goal, obstacle_values):
    if start == goal:
        return [start]
    H, W = len(grid), len(grid[0])
    (sr, sc), (gr, gc) = start, goal
    def h(r, c): return abs(r - gr) + abs(c - gc)
    openpq = []
    heapq.heappush(openpq, (h(sr, sc), 0, (sr, sc)))
    came = { (sr, sc): None }
    g = { (sr, sc): 0 }

    while openpq:
        _, g_curr, (r, c) = heapq.heappop(openpq)
        if (r, c) == (gr, gc):
            path = []
            u = (r, c)
            while u is not None:
                path.append(u)
                u = came[u]
            return path[::-1]
        for dr, dc in FOUR_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, H, W): 
                continue
            if not free_cell(grid, nr, nc, obstacle_values):
                continue
            ng = g_curr + 1
            if (nr, nc) not in g or ng < g[(nr, nc)]:
                g[(nr, nc)] = ng
                f = ng + h(nr, nc)
                came[(nr, nc)] = (r, c)
                heapq.heappush(openpq, (f, ng, (nr, nc)))
    return []

def dfs_coverage(grid, start=None, obstacle_values={1}):
    H, W = len(grid), len(grid[0])
    if start is None:
        for r in range(H):
            for c in range(W):
                if free_cell(grid, r, c, obstacle_values):
                    start = (r, c); break
            if start: break
        if not start: return []
    visited = set([start])
    path = [start]
    stack = [start]
    while stack:
        r, c = stack[-1]
        progressed = False
        for dr, dc in FOUR_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, H, W) and free_cell(grid, nr, nc, obstacle_values) and (nr, nc) not in visited:
                stack.append((nr, nc))
                visited.add((nr, nc))
                path.append((nr, nc))
                progressed = True
                break
        if not progressed:
            stack.pop()
            if stack: path.append(stack[-1])
    return path

def build_supermask(grid, obstacle_values={1}, block=2):
    H, W = len(grid), len(grid[0])
    MR, MC = H // block, W // block
    super_free = [[False]*MC for _ in range(MR)]
    for mr in range(MR):
        for mc in range(MC):
            ok = True
            for i in range(block):
                for j in range(block):
                    r = mr*block + i
                    c = mc*block + j
                    if not free_cell(grid, r, c, obstacle_values):
                        ok = False; break
                if not ok: break
            super_free[mr][mc] = ok
    return super_free

def super_neighbors(mr, mc, MR, MC):
    for dmr, dmc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nmr, nmc = mr + dmr, mc + dmc
        if 0 <= nmr < MR and 0 <= nmc < MC:
            yield nmr, nmc

def super_dfs_order(super_free, start_m=None):
    MR, MC = len(super_free), len(super_free[0])
    if start_m is None:
        for mr in range(MR):
            for mc in range(MC):
                if super_free[mr][mc]:
                    start_m = (mr, mc); break
            if start_m: break
        if start_m is None: return []
    order, seen = [], set()
    def rec(mr, mc):
        seen.add((mr, mc)); order.append((mr, mc))
        for nmr, nmc in super_neighbors(mr, mc, MR, MC):
            if super_free[nmr][nmc] and (nmr, nmc) not in seen:
                rec(nmr, nmc)
    mr0, mc0 = start_m
    if super_free[mr0][mc0]: rec(mr0, mc0)
    else: return []
    return order

def micro_cells_for_block(mr, mc, block=2):
    r0, c0 = mr*block, mc*block
    return [(r0, c0), (r0, c0+1), (r0+1, c0+1), (r0+1, c0)]

def stc_path(grid, start=None, obstacle_values={1}, block=2, mop_up=True):
    H, W = len(grid), len(grid[0])
    if start is None:
        for r in range(H):
            for c in range(W):
                if free_cell(grid, r, c, obstacle_values):
                    start = (r, c); break
            if start: break
        if not start: return []

    super_free = build_supermask(grid, obstacle_values, block=block)
    MR, MC = len(super_free), len(super_free[0])
    mr0, mc0 = start[0] // block, start[1] // block
    start_m = (mr0, mc0) if (0 <= mr0 < MR and 0 <= mc0 < MC) else None
    order = super_dfs_order(super_free, start_m=start_m)
    if not order:
        return dfs_coverage(grid, start=start, obstacle_values=obstacle_values)

    path = []
    cur = start
    visited_cells = set()
    if free_cell(grid, *cur, obstacle_values):
        path.append(cur); visited_cells.add(cur)

    def append_connector(to_cell):
        nonlocal cur, path
        conn = astar(grid, cur, to_cell, obstacle_values)
        if not conn: return False
        if conn and conn[0] == cur: conn = conn[1:]
        path.extend(conn)
        for p in conn: visited_cells.add(p)
        cur = to_cell
        return True

    for mr, mc in order:
        micro = micro_cells_for_block(mr, mc, block=block)
        if cur != micro[0]:
            if not append_connector(micro[0]): continue
        for cell in micro[1:]:
            if not append_connector(cell): break

    if mop_up:
        H, W = len(grid), len(grid[0])
        free_unvisited = [(r, c) for r in range(H) for c in range(W)
                          if free_cell(grid, r, c, obstacle_values) and (r, c) not in visited_cells]
        while free_unvisited:
            target = min(free_unvisited, key=lambda p: abs(p[0]-cur[0]) + abs(p[1]-cur[1]))
            if cur != target: append_connector(target)
            stack, seen_local = [target], {target}
            while stack:
                r, c = stack.pop()
                if cur != (r, c): append_connector((r, c))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc, H, W) and free_cell(grid, nr, nc, obstacle_values):
                        if (nr, nc) not in seen_local and (nr, nc) not in visited_cells:
                            seen_local.add((nr, nc)); stack.append((nr, nc))
            visited_cells.update(seen_local)
            free_unvisited = [(r, c) for (r, c) in free_unvisited if (r, c) not in visited_cells]
    return path

# ===========================
# PYGAME VISUALIZATION
# ===========================
# Color palette
COL_BG         = (245, 245, 245)
COL_GRID       = (210, 210, 210)
COL_OBS        = (50, 50, 55)
COL_SPILL      = (255, 220, 70)
COL_FREE       = (235, 240, 245)
COL_VISITED    = (200, 225, 255)
COL_PATH       = (120, 170, 220)
COL_CURRENT    = (30, 144, 255)
COL_START      = (60, 170, 60)
COL_TEXT       = (30, 30, 30)
COL_SUPEREDGE  = (180, 180, 180)

def auto_cell_size(H, W, max_w=1100, max_h=800, margin=160):
    cw = max(6, (max_w - margin)//W)
    ch = max(6, (max_h - margin)//H)
    return int(max(6, min(cw, ch)))

def draw_grid(screen, grid, cell_sz, offset, obstacle_values, spills_value=None,
              show_grid=True, super_block=2, show_super=True):
    H, W = len(grid), len(grid[0])
    ox, oy = offset
    # cells
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            x = ox + c*cell_sz
            y = oy + r*cell_sz
            if v in obstacle_values:
                color = COL_OBS
            elif spills_value is not None and v == spills_value:
                color = COL_SPILL
            else:
                color = COL_FREE
            pygame.draw.rect(screen, color, (x, y, cell_sz, cell_sz))
    # grid lines
    if show_grid:
        for r in range(H+1):
            y = oy + r*cell_sz
            pygame.draw.line(screen, COL_GRID, (ox, y), (ox + W*cell_sz, y), 1)
        for c in range(W+1):
            x = ox + c*cell_sz
            pygame.draw.line(screen, COL_GRID, (x, oy), (x, oy + H*cell_sz), 1)
    # super-cell boundaries
    if show_super and super_block > 1:
        thick = 2
        for sr in range(0, H+1, super_block):
            y = oy + sr*cell_sz
            pygame.draw.line(screen, COL_SUPEREDGE, (ox, y), (ox + W*cell_sz, y), thick)
        for sc in range(0, W+1, super_block):
            x = ox + sc*cell_sz
            pygame.draw.line(screen, COL_SUPEREDGE, (x, oy), (x, oy + H*cell_sz), thick)

def draw_path(screen, path, idx, cell_sz, offset, visited_set):
    ox, oy = offset
    # visited fill
    for (r, c) in visited_set:
        x = ox + c*cell_sz
        y = oy + r*cell_sz
        pygame.draw.rect(screen, COL_VISITED, (x, y, cell_sz, cell_sz))
    # path polyline
    if len(path) >= 2:
        pts = []
        for (r, c) in path[:max(1, idx+1)]:
            x = ox + c*cell_sz + cell_sz//2
            y = oy + r*cell_sz + cell_sz//2
            pts.append((x, y))
        if len(pts) >= 2:
            pygame.draw.lines(screen, COL_PATH, False, pts, 3)
    # current robot
    r, c = path[idx]
    cx = ox + c*cell_sz + cell_sz//2
    cy = oy + r*cell_sz + cell_sz//2
    rad = max(4, cell_sz//3)
    pygame.draw.circle(screen, COL_CURRENT, (cx, cy), rad)

def run_visualization(grid, start=None, obstacle_values={2}, spills_value=1,
                      super_block=2, mop_up=True):
    H, W = len(grid), len(grid[0])
    if start is None:
        # pick first free cell
        for r in range(H):
            for c in range(W):
                if grid[r][c] not in obstacle_values:
                    start = (r, c); break
            if start: break

    path = stc_path(grid, start=start, obstacle_values=obstacle_values,
                    block=super_block, mop_up=mop_up)
    if not path:
        print("No path found."); return

    pygame.init()
    pygame.display.set_caption("Spanning Tree Coverage (STC) – Pygame Visualization")

    cell_sz = auto_cell_size(H, W)
    PAD_X, PAD_Y = 40, 120
    WIN_W = W*cell_sz + PAD_X*2
    WIN_H = H*cell_sz + PAD_Y*2
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    offset = (PAD_X, PAD_Y)

    idx = 0
    visited = set([path[0]])
    paused = False
    show_grid = True
    show_super = True
    steps_per_sec = 20  # speed

    start_cell = path[0]

    def draw_ui():
        # header
        title = "STC (2x2)  |  cells: {}x{}  |  path len: {}  |  speed: {} cps".format(
            H, W, len(path), steps_per_sec)
        t1 = font.render(title, True, COL_TEXT)
        screen.blit(t1, (20, 20))
        t2 = font.render("Space=Pause  N=Step  +/-=Speed  G=Grid  S=Screenshot  Q/Esc=Quit", True, COL_TEXT)
        screen.blit(t2, (20, 46))

    # Pre-draw static background once (for crisp redraws)
    bg = pygame.Surface((WIN_W, WIN_H))
    bg.fill(COL_BG)
    draw_grid(bg, grid, cell_sz, offset, obstacle_values, spills_value,
              show_grid=True, super_block=super_block, show_super=show_super)
    # Start marker
    sx = offset[0] + start_cell[1]*cell_sz
    sy = offset[1] + start_cell[0]*cell_sz
    pygame.draw.rect(bg, COL_START, (sx+cell_sz*0.2, sy+cell_sz*0.2, cell_sz*0.6, cell_sz*0.6), border_radius=6)

    last_step_time = 0.0
    step_interval = 1.0/max(1, steps_per_sec)

    running = True
    while running:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    if idx < len(path)-1:
                        idx += 1; visited.add(path[idx])
                        last_step_time = now
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    steps_per_sec = min(240, steps_per_sec + 5)
                    step_interval = 1.0/max(1, steps_per_sec)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    steps_per_sec = max(1, steps_per_sec - 5)
                    step_interval = 1.0/max(1, steps_per_sec)
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_s:
                    fname = f"stc_frame_{int(time.time())}.png"
                    pygame.image.save(screen, fname)
                    print(f"Saved screenshot: {fname}")

        if not paused and idx < len(path)-1 and (now - last_step_time) >= step_interval:
            idx += 1; visited.add(path[idx]); last_step_time = now

        # redraw static bg each frame to respect grid toggle
        screen.blit(bg, (0,0))
        if not show_grid or not show_super:
            # re-draw grid with toggles applied
            draw_grid(screen, grid, cell_sz, offset, obstacle_values, spills_value,
                      show_grid=show_grid, super_block=super_block, show_super=show_super)
            # re-draw start marker
            pygame.draw.rect(screen, COL_START, (sx+cell_sz*0.2, sy+cell_sz*0.2, cell_sz*0.6, cell_sz*0.6),
                             border_radius=6)

        draw_path(screen, path, idx, cell_sz, offset, visited)
        draw_ui()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# ===========================
# Example map
# 0 = free, 1 = chemical spill (traversable, just colored), 2 = obstacle (blocked)
# Change obstacle_values={2} if you use this encoding.
# ===========================
if __name__ == "__main__":
    grid_map = [
        [0,0,0,0,0,0,0,0,0,0,2,2,0,0,0,0,0,0],
        [0,1,1,0,0,0,2,2,0,0,2,2,0,0,1,1,0,0],
        [0,1,1,0,0,0,2,2,0,0,0,0,0,0,1,1,0,0],
        [0,0,0,0,2,0,0,0,0,2,0,0,0,0,0,0,2,0],
        [0,2,2,0,2,0,0,1,1,2,0,0,2,2,0,0,2,0],
        [0,0,0,0,2,0,0,1,1,2,0,0,2,2,0,0,0,0],
        [0,0,0,0,0,0,2,2,0,0,0,2,2,0,0,0,0,0],
        [2,2,0,0,0,0,2,2,0,0,0,2,2,0,0,0,0,0],
        [0,0,0,1,1,0,0,0,0,2,0,0,0,1,1,0,0,0],
        [0,0,0,1,1,0,0,0,0,2,0,0,0,1,1,0,0,0],
    ]

    start = (0, 0)             # starting cell (row, col)
    obstacle_vals = {2}        # treat only "2" as obstacle; "1" is a spill (drawn yellow)
    super_block = 2            # classic STC is 2x2
    mop_up = True              # sweep narrow pockets after STC

    run_visualization(grid_map, start=start, obstacle_values=obstacle_vals,
                      spills_value=1, super_block=super_block, mop_up=mop_up)
