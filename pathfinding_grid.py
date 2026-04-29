"""
Grid Pathfinding Visualization: Bellman-Ford vs Dijkstra
=========================================================
Generates animated GIFs showing both algorithms navigating a 2-D grid maze.
Each cell is either open (passable) or a wall. The algorithms find the
shortest path from the top-left to the bottom-right corner.

Outputs:
  grid_dijkstra_small.gif   — 20×20 grid, Dijkstra
  grid_bf_small.gif         — 20×20 grid, Bellman-Ford
  grid_comparison_small.gif — 20×20 side-by-side
  grid_dijkstra_large.gif   — 40×40 grid, Dijkstra
  grid_bf_large.gif         — 40×40 grid, Bellman-Ford
  grid_comparison_large.gif — 40×40 side-by-side
"""

import random
import heapq
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from collections import deque

random.seed(7)

# ─────────────────────────────────────────────
# COLOUR PALETTE  (matches main script theme)
# ─────────────────────────────────────────────
BG = "#0f1117"
PANEL = "#1a1d27"
ACCENT = "#4fc3f7"  # Dijkstra frontier — sky blue
ACCENT2 = "#ef5350"  # Bellman-Ford frontier — red
GREEN = "#66bb6a"  # settled / finalized
YELLOW = "#ffd54f"  # currently processing
GREY = "#546e7a"  # walls
WHITE = "#eceff1"  # open cells
GOLD = "#ffca28"  # shortest path
DARK = "#0d1117"  # background behind grid

# Cell state codes used in the render array
WALL = 0
OPEN = 1
FRONTIER = 2  # reached but not settled
SETTLED = 3  # permanently finalised
CURRENT = 4  # the cell being processed this frame
PATH = 5  # final shortest path
START = 6
END = 7

# Map state codes to RGBA colours
CELL_COLORS = {
    WALL: np.array([0.06, 0.07, 0.09, 1.0]),  # near-black
    OPEN: np.array([0.10, 0.11, 0.15, 1.0]),  # dark panel
    FRONTIER: np.array([0.31, 0.76, 0.97, 0.55]),  # blue tint
    SETTLED: np.array([0.40, 0.73, 0.42, 0.60]),  # green tint
    CURRENT: np.array([1.00, 0.84, 0.31, 1.00]),  # yellow
    PATH: np.array([1.00, 0.79, 0.16, 1.00]),  # gold
    START: np.array([0.31, 0.76, 0.97, 1.00]),  # bright blue
    END: np.array([0.94, 0.33, 0.31, 1.00]),  # bright red
}

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "text.color": WHITE,
        "font.family": "monospace",
    }
)

# ─────────────────────────────────────────────
# MAZE GENERATION  (recursive back-tracker)
# ─────────────────────────────────────────────


def generate_maze(rows, cols, wall_bias=0.28, seed=None):
    """
    Generates a random maze using a mix of recursive back-tracking
    (guarantees connectivity) and random wall sprinkling (adds shortcuts).
    Returns a 2-D list: 0=wall, 1=open.
    Also guarantees start (0,0) and end (rows-1, cols-1) are open.
    """
    if seed is not None:
        random.seed(seed)

    # Start fully walled
    grid = [[0] * cols for _ in range(rows)]

    # Recursive back-tracker — carves passages on ODD cells
    def carve(r, c):
        grid[r][c] = 1
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                grid[r + dr // 2][c + dc // 2] = 1  # knock down wall between
                carve(nr, nc)

    # Carve from (0,0); maze works best on odd dimensions
    carve(0, 0)

    # Randomly open extra walls to create shortcuts (makes it look richer)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and random.random() < wall_bias:
                grid[r][c] = 1

    # Always ensure start/end are open
    grid[0][0] = 1
    grid[rows - 1][cols - 1] = 1

    return grid


def neighbors_4(r, c, rows, cols, grid):
    """4-connected passable neighbours."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
            yield nr, nc


def trace_path(prev, start, end):
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path if path and path[0] == start else []


# ─────────────────────────────────────────────
# ALGORITHMS — yield snapshots for animation
# ─────────────────────────────────────────────


def run_dijkstra(grid, start, end):
    """
    Yields (dist_dict, settled_set, current_cell) at each step.
    Final yield also returns the path via a sentinel.
    """
    rows = len(grid)
    cols = len(grid[0])
    dist = {start: 0}
    prev = {}
    settled = set()
    heap = [(0, start)]
    snapshots = []  # list of (dist copy, settled copy, current)

    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        snapshots.append((dict(dist), set(settled), u))
        if u == end:
            break
        for v in neighbors_4(u[0], u[1], rows, cols, grid):
            nd = d + 1
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    path = trace_path(prev, start, end)
    return snapshots, path


def run_bellman_ford(grid, start, end):
    """
    Grid Bellman-Ford: runs V-1 passes over all open cells' edges.
    Yields a snapshot after each full pass (not each edge), keeping
    frame count manageable. Stops early if no update occurred.
    """
    rows = len(grid)
    cols = len(grid[0])
    open_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    V = len(open_cells)

    dist = {cell: float("inf") for cell in open_cells}
    dist[start] = 0
    prev = {}
    snapshots = []

    # Track which cells have their final distance already
    # (approximate 'settled' — cells that haven't changed last pass)
    prev_dist = None

    for iteration in range(V - 1):
        updated_this_pass = set()
        changed = False
        for u in open_cells:
            for v in neighbors_4(u[0], u[1], rows, cols, grid):
                if dist[u] + 1 < dist.get(v, float("inf")):
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    updated_this_pass.add(v)
                    changed = True

        # Approximate settled: cells not updated this pass and already reached
        settled_approx = {
            c
            for c in open_cells
            if dist[c] < float("inf") and c not in updated_this_pass
        }

        # One snapshot per pass — current = the cell most recently improved
        current = next(iter(updated_this_pass)) if updated_this_pass else end
        snapshots.append((dict(dist), settled_approx, current))

        if not changed:
            break

    path = trace_path(prev, start, end)
    return snapshots, path


# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────


def build_rgba(grid, dist, settled, current, path=None, start=None, end=None):
    """Convert algorithm state to an RGBA image array."""
    rows = len(grid)
    cols = len(grid[0])
    img = np.zeros((rows, cols, 4))

    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if grid[r][c] == 0:
                img[r, c] = CELL_COLORS[WALL]
            elif path and cell in path:
                img[r, c] = CELL_COLORS[PATH]
            elif cell == current:
                img[r, c] = CELL_COLORS[CURRENT]
            elif cell == start:
                img[r, c] = CELL_COLORS[START]
            elif cell == end:
                img[r, c] = CELL_COLORS[END]
            elif cell in settled:
                img[r, c] = CELL_COLORS[SETTLED]
            elif dist.get(cell, float("inf")) < float("inf"):
                img[r, c] = CELL_COLORS[FRONTIER]
            else:
                img[r, c] = CELL_COLORS[OPEN]

    return img


def draw_grid_frame(
    ax,
    grid,
    dist,
    settled,
    current,
    title,
    step,
    path=None,
    start=None,
    end=None,
    path_len=None,
    steps_total=None,
):
    ax.clear()
    ax.set_facecolor(DARK)

    path_set = set(map(tuple, path)) if path else set()
    img = build_rgba(grid, dist, settled, current, path=path_set, start=start, end=end)

    ax.imshow(img, interpolation="nearest", aspect="equal")

    # Mark start/end with letters
    if start:
        ax.text(
            start[1],
            start[0],
            "S",
            color=DARK,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
        )
    if end:
        ax.text(
            end[1],
            end[0],
            "E",
            color=DARK,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Title + stats
    stats = f"Step {step}"
    if path_len is not None:
        stats += f"  |  Path length: {path_len}"
    if steps_total is not None:
        stats += f"  |  Total steps: {steps_total}"
    ax.set_title(f"{title}\n{stats}", color=WHITE, fontsize=9, fontweight="bold", pad=6)
    ax.axis("off")


# ─────────────────────────────────────────────
# GIF SAVERS
# ─────────────────────────────────────────────


def save_single_gif(
    grid, snapshots, path, start, end, filename, title, max_frames=180, fps=12
):
    """Save one algorithm's traversal as a GIF."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08)

    # Sub-sample frames if needed
    if len(snapshots) > max_frames:
        step = max(1, len(snapshots) // max_frames)
        frames_to_use = snapshots[::step]
    else:
        frames_to_use = snapshots

    # Add final frame with path highlighted (held for longer)
    last_dist, last_settled, _ = snapshots[-1]
    path_set = set(map(tuple, path))
    final_settled = last_settled | path_set
    HOLD_FRAMES = 18  # freeze on final path for ~1.5s

    legend_elements = [
        mpatches.Patch(
            color=np.array(CELL_COLORS[CURRENT][:3]), label="Currently processing"
        ),
        mpatches.Patch(
            color=np.array(CELL_COLORS[SETTLED][:3]), label="Settled / explored"
        ),
        mpatches.Patch(
            color=np.array(CELL_COLORS[FRONTIER][:3]), label="Frontier (reached)"
        ),
        mpatches.Patch(color=np.array(CELL_COLORS[PATH][:3]), label="Shortest path"),
        mpatches.Patch(color=np.array(CELL_COLORS[WALL][:3]), label="Wall"),
    ]

    total_frames = len(frames_to_use) + HOLD_FRAMES

    def update(frame):
        if frame < len(frames_to_use):
            dist, settled, current = frames_to_use[frame]
            draw_grid_frame(
                ax,
                grid,
                dist,
                settled,
                current,
                title,
                step=frame + 1,
                start=start,
                end=end,
            )
        else:
            # Final: show path
            draw_grid_frame(
                ax,
                grid,
                last_dist,
                final_settled,
                None,
                title,
                step=len(frames_to_use),
                path=path,
                start=start,
                end=end,
                path_len=len(path) - 1,
                steps_total=len(snapshots),
            )
        ax.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=3,
            fontsize=6.5,
            facecolor=PANEL,
            edgecolor=GREY,
            labelcolor=WHITE,
            framealpha=0.9,
        )

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 // fps)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(
        f"  Saved: {filename}  ({len(snapshots)} algo steps, "
        f"{total_frames} frames, path={len(path) - 1 if path else 'N/A'})"
    )


def save_comparison_gif(
    grid,
    snaps_dj,
    path_dj,
    snaps_bf,
    path_bf,
    start,
    end,
    filename,
    size_label,
    max_frames=180,
    fps=10,
):
    """Side-by-side comparison GIF — both algorithms on the same maze."""
    fig, (ax_dj, ax_bf) = plt.subplots(1, 2, figsize=(10, 5.2))
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.10, wspace=0.06)

    n_dj = len(snaps_dj)
    n_bf = len(snaps_bf)
    total = max(n_dj, n_bf)

    if total > max_frames:
        step = max(1, total // max_frames)
        idx_dj = [min(i, n_dj - 1) for i in range(0, total, step)]
        idx_bf = [min(i, n_bf - 1) for i in range(0, total, step)]
    else:
        idx_dj = [min(i, n_dj - 1) for i in range(total)]
        idx_bf = [min(i, n_bf - 1) for i in range(total)]

    HOLD = 20
    total_frames = len(idx_dj) + HOLD

    last_dj = snaps_dj[-1]
    last_bf = snaps_bf[-1]
    path_set_dj = set(map(tuple, path_dj))
    path_set_bf = set(map(tuple, path_bf))

    legend_elements = [
        mpatches.Patch(color=np.array(CELL_COLORS[CURRENT][:3]), label="Processing"),
        mpatches.Patch(color=np.array(CELL_COLORS[SETTLED][:3]), label="Settled"),
        mpatches.Patch(color=np.array(CELL_COLORS[FRONTIER][:3]), label="Frontier"),
        mpatches.Patch(color=np.array(CELL_COLORS[PATH][:3]), label="Path"),
        mpatches.Patch(color=np.array(CELL_COLORS[WALL][:3]), label="Wall"),
    ]

    fig.suptitle(
        f"Grid Pathfinding Comparison  [{size_label}]",
        color=WHITE,
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    def update(frame):
        if frame < len(idx_dj):
            i_dj = idx_dj[frame]
            i_bf = idx_bf[frame]
            d_dj, s_dj, c_dj = snaps_dj[i_dj]
            d_bf, s_bf, c_bf = snaps_bf[i_bf]
            draw_grid_frame(
                ax_dj,
                grid,
                d_dj,
                s_dj,
                c_dj,
                "Dijkstra's  (Greedy)",
                step=i_dj + 1,
                start=start,
                end=end,
            )
            draw_grid_frame(
                ax_bf,
                grid,
                d_bf,
                s_bf,
                c_bf,
                "Bellman-Ford  (Dynamic Programming)",
                step=i_bf + 1,
                start=start,
                end=end,
            )
        else:
            # Hold on final path frames
            d_dj, s_dj, _ = last_dj
            d_bf, s_bf, _ = last_bf
            draw_grid_frame(
                ax_dj,
                grid,
                d_dj,
                s_dj | path_set_dj,
                None,
                "Dijkstra's  (Greedy)",
                step=n_dj,
                path=path_dj,
                start=start,
                end=end,
                path_len=len(path_dj) - 1,
                steps_total=n_dj,
            )
            draw_grid_frame(
                ax_bf,
                grid,
                d_bf,
                s_bf | path_set_bf,
                None,
                "Bellman-Ford  (Dynamic Programming)",
                step=n_bf,
                path=path_bf,
                start=start,
                end=end,
                path_len=len(path_bf) - 1,
                steps_total=n_bf,
            )

        for ax in [ax_dj, ax_bf]:
            ax.legend(
                handles=legend_elements,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=5,
                fontsize=6,
                facecolor=PANEL,
                edgecolor=GREY,
                labelcolor=WHITE,
                framealpha=0.9,
            )

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 // fps)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def run():
    configs = [
        # (rows, cols, wall_bias, seed, label)
        (21, 21, 0.22, 3, "small", "21×21"),
        (31, 31, 0.20, 7, "large", "31×31"),
    ]

    for rows, cols, wb, seed, tag, size_label in configs:
        print(f"\n=== {size_label} grid ===")

        grid = generate_maze(rows, cols, wall_bias=wb, seed=seed)
        start = (0, 0)
        end = (rows - 1, cols - 1)

        print(f"  Running Dijkstra...")
        snaps_dj, path_dj = run_dijkstra(grid, start, end)

        print(f"  Running Bellman-Ford...")
        snaps_bf, path_bf = run_bellman_ford(grid, start, end)

        print(
            f"  Dijkstra: {len(snaps_dj)} steps | "
            f"BF: {len(snaps_bf)} passes | "
            f"Path length: {len(path_dj) - 1 if path_dj else 'N/A'}"
        )

        # Individual GIFs
        save_single_gif(
            grid,
            snaps_dj,
            path_dj,
            start,
            end,
            f"grid_dijkstra_{tag}.gif",
            f"Dijkstra's Algorithm  [{size_label}]",
            max_frames=120,
            fps=12,
        )

        save_single_gif(
            grid,
            snaps_bf,
            path_bf,
            start,
            end,
            f"grid_bf_{tag}.gif",
            f"Bellman-Ford Algorithm  [{size_label}]",
            max_frames=80,
            fps=8,
        )

        # Side-by-side comparison
        save_comparison_gif(
            grid,
            snaps_dj,
            path_dj,
            snaps_bf,
            path_bf,
            start,
            end,
            f"grid_comparison_{tag}.gif",
            size_label,
            max_frames=100,
            fps=10,
        )

    print("\n✓ Done. Files produced:")
    print("  grid_dijkstra_small.gif   grid_bf_small.gif   grid_comparison_small.gif")
    print("  grid_dijkstra_large.gif   grid_bf_large.gif   grid_comparison_large.gif")


if __name__ == "__main__":
    run()
