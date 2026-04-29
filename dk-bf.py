"""
Graph Algorithms Visualization: Bellman-Ford vs. Dijkstra (Enhanced)
=====================================================================
Outputs produced by run_demo():
  bf_10.gif                    — Bellman-Ford step animation (settled nodes green,
                                 path highlighted gold on final frame, visit-order labels)
  dj_10.gif                    — Dijkstra step animation (same extras)
  comparison_10.gif            — Side-by-side on identical graph
  negative_weight_comparison.png — Correctness test (neg weights, Dijkstra provably wrong)
  timing_growth.gif            — Animated runtime bar chart
  complexity_table.png         — Theory vs experiment summary slide
  convergence_curves.png       — Distance-to-target over steps, both algorithms
  memory_usage.png             — Peak memory bar chart per graph size
"""

import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import random
import time
import heapq
import tracemalloc
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
import math
from collections import deque

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
BG = "#0f1117"
PANEL = "#1a1d27"
ACCENT = "#4fc3f7"  # sky-blue  — Dijkstra
ACCENT2 = "#ef5350"  # red       — Bellman-Ford / wrong
GREEN = "#66bb6a"  # settled / correct
YELLOW = "#ffd54f"  # currently processing / path highlight
GREY = "#546e7a"  # unvisited / muted
WHITE = "#eceff1"
GOLD = "#ffca28"  # shortest-path edge highlight

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GREY,
        "axes.labelcolor": WHITE,
        "xtick.color": WHITE,
        "ytick.color": WHITE,
        "text.color": WHITE,
        "font.family": "monospace",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    }
)

# ─────────────────────────────────────────────
# GRAPH GENERATION
# ─────────────────────────────────────────────


def ensure_reachability(G, start, target):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == target:
            return G
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    available = list(set(G.nodes()) - {start, target})
    random.shuffle(available)
    path = [start] + available[: random.randint(1, min(3, len(available)))] + [target]
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=random.randint(1, 10))
    return G


def generate_graph(n, edge_factor=1, max_edges=None):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    num_edges = (
        n * edge_factor if max_edges is None else min(n * edge_factor, max_edges)
    )
    added = 0
    while added < num_edges:
        u, v = random.randint(0, n - 1), random.randint(0, n - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=random.randint(1, 10))
            added += 1
    G = ensure_reachability(G, 0, n - 1)
    return G


def grid_layout(G):
    n = len(G.nodes())
    cols = max(1, int(math.ceil(math.sqrt(n))))
    pos = {}
    for i, node in enumerate(sorted(G.nodes())):
        pos[node] = (i % cols, -(i // cols))
    return pos


# ─────────────────────────────────────────────
# PATH TRACING
# ─────────────────────────────────────────────


def trace_path(prev, start, target):
    """Reconstruct the shortest path from the prev-node dictionary."""
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    if not path or path[0] != start:
        return []
    return path


# ─────────────────────────────────────────────
# ALGORITHMS  (return prev + visited_order)
# ─────────────────────────────────────────────


def bellman_ford(G, start, target, record_steps=False):
    """
    Returns (dist, prev, steps, currents, iteration_nums, visited_order).
    visited_order maps node -> the pass number it was first reached.
    """
    dist = {node: float("inf") for node in G.nodes()}
    dist[start] = 0
    prev = {node: None for node in G.nodes()}
    visited_order = {start: 0}
    steps, currents, iteration_nums = (
        ([], [], []) if record_steps else (None, None, None)
    )
    iteration = 0

    for _ in range(len(G.nodes()) - 1):
        iteration += 1
        for u, v, data in G.edges(data=True):
            w = data["weight"]
            for a, b in [(u, v), (v, u)]:
                if dist[a] + w < dist[b]:
                    dist[b] = dist[a] + w
                    prev[b] = a
                    if b not in visited_order:
                        visited_order[b] = iteration
                if record_steps:
                    steps.append(dist.copy())
                    currents.append(a)
                    iteration_nums.append(iteration)

    return dist, prev, steps, currents, iteration_nums, visited_order


def dijkstra(G, start, target, record_steps=False):
    """
    Returns (dist, prev, steps, currents, iteration_nums, visited_order, settled_sets).
    settled_sets[i] is the frozenset of settled nodes at step i.
    visited_order maps node -> the order it was popped from the heap.
    """
    dist = {node: float("inf") for node in G.nodes()}
    dist[start] = 0
    prev = {node: None for node in G.nodes()}
    visited = set()
    visited_order = {}
    settle_count = 0
    steps, currents, iteration_nums, settled_sets = (
        ([], [], [], []) if record_steps else (None, None, None, None)
    )
    heap = [(0, start)]
    iteration = 0

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        settle_count += 1
        visited_order[u] = settle_count
        iteration += 1
        for v in G.neighbors(u):
            w = G[u][v]["weight"]
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
                if record_steps:
                    steps.append(dist.copy())
                    currents.append(u)
                    iteration_nums.append(iteration)
                    settled_sets.append(frozenset(visited))

    return dist, prev, steps, currents, iteration_nums, visited_order, settled_sets


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────


def node_colors(G, dist, current, settled):
    """
    settled : set of permanently finalized nodes  -> GREEN
    current : node being processed right now       -> YELLOW
    reached : dist < inf but not settled           -> ACCENT (blue)
    other   : not yet reached                      -> GREY
    """
    colors = []
    for node in G.nodes():
        if node == current:
            colors.append(YELLOW)
        elif node in settled:
            colors.append(GREEN)
        elif dist[node] < float("inf"):
            colors.append(ACCENT)
        else:
            colors.append(GREY)
    return colors


def draw_frame(
    G,
    pos,
    dist,
    current,
    ax,
    start,
    target,
    title="",
    step_num=None,
    iteration=None,
    settled=None,
    visit_order=None,
    path_edges=None,
    is_final=False,
):
    """
    Draw one animation frame.
      settled     : set of permanently settled nodes (shown green)
      visit_order : dict node -> settlement number (shown as small label, final frame only)
      path_edges  : list of (u,v) tuples highlighted gold (final frame only)
      is_final    : appends '| DONE' to title and activates path/order overlays
    """
    if settled is None:
        settled = set()
    ax.clear()
    ax.set_facecolor(PANEL)

    colors = node_colors(G, dist, current, settled)
    edge_labels = nx.get_edge_attributes(G, "weight")

    # Edges — highlight shortest path gold on final frame
    if path_edges and is_final:
        path_set = set(map(frozenset, path_edges))
        ec = [GOLD if frozenset([u, v]) in path_set else GREY for u, v in G.edges()]
        ew = [3.0 if frozenset([u, v]) in path_set else 1.2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ec, width=ew, alpha=0.9)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GREY, width=1.2, alpha=0.6)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=colors,
        node_size=600,
        linewidths=1.5,
        edgecolors=WHITE,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_color=BG, font_size=8, font_weight="bold"
    )

    if len(G.nodes()) <= 12:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=7,
            font_color=WHITE,
            bbox=dict(boxstyle="round,pad=0.1", fc=PANEL, ec="none", alpha=0.7),
        )

    # Visit-order numbers — small badge offset from each node, final frame only
    if visit_order and is_final:
        for node, order in visit_order.items():
            if node in pos:
                x, y = pos[node]
                ax.text(
                    x + 0.28,
                    y + 0.28,
                    str(order),
                    fontsize=6,
                    color=YELLOW,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc=BG, ec="none", alpha=0.75),
                )

    header = title
    if step_num is not None:
        header += f"  |  Step {step_num}"
    if iteration is not None:
        header += f"  |  Pass {iteration}"
    if is_final:
        header += "  |  DONE"
    ax.set_title(header, color=WHITE, pad=8)

    # Distance readout — clipped so it doesn't overflow
    finite = {n: d for n, d in dist.items() if d < float("inf")}
    dist_pairs = "  ".join(f"{n}:{d}" for n, d in sorted(finite.items()))
    legend_text = "Distances: " + (
        dist_pairs[:80] + "…" if len(dist_pairs) > 80 else dist_pairs
    )
    ax.text(
        0.01,
        -0.07,
        legend_text,
        transform=ax.transAxes,
        fontsize=6.5,
        color=ACCENT,
        va="top",
    )

    # Start/target — bottom-right, away from title
    ax.text(
        0.99,
        -0.07,
        f"Start: {start}  →  Target: {target}",
        transform=ax.transAxes,
        fontsize=7.5,
        color=GREY,
        ha="right",
        va="top",
    )

    ax.axis("off")


# ─────────────────────────────────────────────
# GIF 1 — Bellman-Ford individual
# ─────────────────────────────────────────────


def save_gif_bf(
    G,
    pos,
    steps,
    currents,
    iteration_nums,
    final_dist,
    prev,
    visit_order,
    filename,
    start,
    target,
):
    """
    BF has no hard 'settled' set mid-run, so we approximate:
    a node is considered converged when its distance already equals
    its final answer. On the last frame the path is drawn in gold
    and visit-order badges are shown.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)

    MAX_FRAMES = 200
    if len(steps) > MAX_FRAMES:
        idx = list(range(0, len(steps), max(1, len(steps) // MAX_FRAMES)))
        steps = [steps[i] for i in idx]
        currents = [currents[i] for i in idx]
        iteration_nums = [iteration_nums[i] for i in idx]

    path_nodes = trace_path(prev, start, target)
    path_edges = list(zip(path_nodes, path_nodes[1:])) if len(path_nodes) > 1 else []

    legend_elements = [
        mpatches.Patch(color=YELLOW, label="Currently processing"),
        mpatches.Patch(color=GREEN, label="Converged to final value"),
        mpatches.Patch(color=ACCENT, label="Reached (tentative)"),
        mpatches.Patch(color=GREY, label="Not yet reached"),
        mlines.Line2D([], [], color=GOLD, linewidth=2.5, label="Shortest path"),
    ]

    total_frames = len(steps)

    def update(frame):
        is_final = frame == total_frames - 1
        dist = steps[frame]
        settled = {
            n for n in G.nodes() if dist[n] == final_dist[n] and dist[n] < float("inf")
        }
        draw_frame(
            G,
            pos,
            dist,
            currents[frame],
            ax,
            start,
            target,
            title="Bellman-Ford (Dynamic Programming)",
            step_num=frame + 1,
            iteration=iteration_nums[frame],
            settled=settled,
            visit_order=visit_order if is_final else None,
            path_edges=path_edges if is_final else None,
            is_final=is_final,
        )
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=7,
            facecolor=PANEL,
            edgecolor=GREY,
            labelcolor=WHITE,
        )

    anim = FuncAnimation(fig, update, frames=total_frames, interval=300)
    anim.save(filename, writer=PillowWriter(fps=3))
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# GIF 2 — Dijkstra individual
# ─────────────────────────────────────────────


def save_gif_dj(
    G,
    pos,
    steps,
    currents,
    iteration_nums,
    settled_sets,
    final_dist,
    prev,
    visit_order,
    filename,
    start,
    target,
):
    """
    Uses the real settled set captured during the algorithm run.
    Final frame highlights the shortest path gold and adds visit-order badges.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)

    MAX_FRAMES = 200
    if len(steps) > MAX_FRAMES:
        idx = list(range(0, len(steps), max(1, len(steps) // MAX_FRAMES)))
        steps = [steps[i] for i in idx]
        currents = [currents[i] for i in idx]
        iteration_nums = [iteration_nums[i] for i in idx]
        settled_sets = [settled_sets[i] for i in idx]

    path_nodes = trace_path(prev, start, target)
    path_edges = list(zip(path_nodes, path_nodes[1:])) if len(path_nodes) > 1 else []

    legend_elements = [
        mpatches.Patch(color=YELLOW, label="Currently processing"),
        mpatches.Patch(color=GREEN, label="Settled / finalized"),
        mpatches.Patch(color=ACCENT, label="Reached (tentative)"),
        mpatches.Patch(color=GREY, label="Not yet reached"),
        mlines.Line2D([], [], color=GOLD, linewidth=2.5, label="Shortest path"),
    ]

    total_frames = len(steps)

    def update(frame):
        is_final = frame == total_frames - 1
        draw_frame(
            G,
            pos,
            steps[frame],
            currents[frame],
            ax,
            start,
            target,
            title="Dijkstra's Algorithm (Greedy)",
            step_num=frame + 1,
            iteration=iteration_nums[frame],
            settled=set(settled_sets[frame]),
            visit_order=visit_order if is_final else None,
            path_edges=path_edges if is_final else None,
            is_final=is_final,
        )
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=7,
            facecolor=PANEL,
            edgecolor=GREY,
            labelcolor=WHITE,
        )

    anim = FuncAnimation(fig, update, frames=total_frames, interval=300)
    anim.save(filename, writer=PillowWriter(fps=3))
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# GIF 3 — Side-by-side comparison
# ─────────────────────────────────────────────


def save_comparison_gif(
    G,
    pos,
    bf_steps,
    bf_currents,
    bf_iters,
    bf_final,
    dj_steps,
    dj_currents,
    dj_iters,
    dj_settled,
    filename,
    start,
    target,
):
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(1, 2, figure=fig, wspace=0.05)
    ax_bf = fig.add_subplot(gs[0])
    ax_dj = fig.add_subplot(gs[1])

    MAX_FRAMES = 200
    n_bf = len(bf_steps)
    n_dj = len(dj_steps)
    total = max(n_bf, n_dj)

    if total > MAX_FRAMES:
        step = max(1, total // MAX_FRAMES)
        idx_bf = [min(i, n_bf - 1) for i in range(0, total, step)]
        idx_dj = [min(i, n_dj - 1) for i in range(0, total, step)]
    else:
        idx_bf = [min(i, n_bf - 1) for i in range(total)]
        idx_dj = [min(i, n_dj - 1) for i in range(total)]

    legend_elements = [
        mpatches.Patch(color=YELLOW, label="Current"),
        mpatches.Patch(color=GREEN, label="Settled"),
        mpatches.Patch(color=ACCENT, label="Reached"),
        mpatches.Patch(color=GREY, label="Unvisited"),
    ]

    def update(frame):
        i_bf = idx_bf[frame]
        i_dj = idx_dj[frame]

        bf_settled = {
            n
            for n in G.nodes()
            if bf_steps[i_bf][n] == bf_final[n] and bf_steps[i_bf][n] < float("inf")
        }

        draw_frame(
            G,
            pos,
            bf_steps[i_bf],
            bf_currents[i_bf],
            ax_bf,
            start,
            target,
            title="Bellman-Ford (Dynamic Programming)",
            step_num=i_bf + 1,
            iteration=bf_iters[i_bf],
            settled=bf_settled,
        )
        draw_frame(
            G,
            pos,
            dj_steps[i_dj],
            dj_currents[i_dj],
            ax_dj,
            start,
            target,
            title="Dijkstra's (Greedy)",
            step_num=i_dj + 1,
            iteration=dj_iters[i_dj],
            settled=set(dj_settled[i_dj]),
        )
        for ax in [ax_bf, ax_dj]:
            ax.legend(
                handles=legend_elements,
                loc="lower right",
                fontsize=7,
                facecolor=PANEL,
                edgecolor=GREY,
                labelcolor=WHITE,
            )

    anim = FuncAnimation(fig, update, frames=len(idx_bf), interval=300)
    anim.save(filename, writer=PillowWriter(fps=3))
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# PNG 4 — Negative weight correctness test
# ─────────────────────────────────────────────


def negative_weight_demo():
    """
    Directed graph that PROVABLY breaks Dijkstra.

    Topology:
        0 --1--> A --1--> C
        0 --2--> B
        B --(-5)--> A   (negative edge)

    True shortest:  0->B->A->C = 2 + (-5) + 1 = -2
    Dijkstra:       settles A at 1, then sees B->A but skips it (A already settled)
                    => A stays at 1, C stays at 2  (WRONG)
    """
    G = nx.DiGraph()
    for u, v, w in [(0, "A", 1), (0, "B", 2), ("B", "A", -5), ("A", "C", 1)]:
        G.add_edge(u, v, weight=w)

    pos = {0: (0, 1), "A": (1.6, 2), "B": (1.6, 0), "C": (3.2, 2)}

    def bf_directed(G, start):
        dist = {n: float("inf") for n in G.nodes()}
        dist[start] = 0
        for _ in range(len(G.nodes()) - 1):
            for u, v, data in G.edges(data=True):
                if dist[u] + data["weight"] < dist[v]:
                    dist[v] = dist[u] + data["weight"]
        return dist

    def dj_directed(G, start):
        dist = {n: float("inf") for n in G.nodes()}
        dist[start] = 0
        visited = set()
        heap = [(0, start)]
        while heap:
            d_u, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            for v in G.successors(u):
                w = G[u][v]["weight"]
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(heap, (dist[v], v))
        return dist

    bf_dist = bf_directed(G, 0)
    dj_dist = dj_directed(G, 0)
    wrong_nodes = [n for n in G.nodes() if dj_dist[n] != bf_dist[n]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Negative Weight Graph: Correctness Comparison",
        color=WHITE,
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    def draw_directed(ax, dist, title, is_correct):
        ax.set_facecolor(PANEL)
        colors = []
        for n in G.nodes():
            if n in wrong_nodes and not is_correct:
                colors.append(ACCENT2)
            elif dist[n] == bf_dist[n] and dist[n] < float("inf"):
                colors.append(GREEN)
            elif dist[n] < float("inf"):
                colors.append(ACCENT)
            else:
                colors.append(GREY)

        edge_colors = [ACCENT2 if G[u][v]["weight"] < 0 else GREY for u, v in G.edges()]
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=22,
            width=2.2,
            connectionstyle="arc3,rad=0.12",
            node_size=1000,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=colors,
            node_size=1000,
            linewidths=2,
            edgecolors=WHITE,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_color=BG, font_size=12, font_weight="bold"
        )
        edge_labels = {(u, v): str(data["weight"]) for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=9,
            font_color=YELLOW,
            bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec="none", alpha=0.85),
        )
        dist_lines = "\n".join(
            f"Node {n}:  {d if d < float('inf') else 'inf'}"
            + (" ✗ WRONG" if n in wrong_nodes and not is_correct else "")
            for n, d in sorted(dist.items(), key=lambda x: str(x[0]))
        )
        ax.text(
            0.04,
            0.04,
            dist_lines,
            transform=ax.transAxes,
            fontsize=9,
            color=WHITE,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=GREY, alpha=0.9),
        )
        status_color = GREEN if is_correct else ACCENT2
        status_text = (
            "✓ CORRECT" if is_correct else "✗ WRONG — misses negative shortcut"
        )
        ax.set_title(f"{title}\n{status_text}", color=status_color, pad=8, fontsize=10)
        ax.axis("off")

    draw_directed(axes[0], bf_dist, "Bellman-Ford (Dynamic Programming)", True)
    draw_directed(
        axes[1], dj_dist, "Dijkstra's Algorithm (Greedy)", len(wrong_nodes) == 0
    )

    note = (
        "Red edge (B→A, weight −5):  true shortest  0→B→A→C  =  2 + (−5) + 1  =  −2\n"
        "Dijkstra pops A first (cost 1 via 0→A) and FINALIZES it. When B is later popped\n"
        "and B→A is seen, A is already settled — skipped entirely. A stays at 1, C stays at 2.\n"
        "Bellman-Ford re-relaxes all edges V−1 times, eventually propagating the correction."
    )
    fig.text(
        0.5,
        -0.05,
        note,
        ha="center",
        va="top",
        fontsize=8.5,
        color=WHITE,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.7", fc=PANEL, ec=GREY, alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(
        "negative_weight_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG
    )
    plt.close(fig)
    print("  Saved: negative_weight_comparison.png")
    print(f"  BF: {bf_dist}  |  DJ: {dj_dist}  |  Wrong nodes: {wrong_nodes}")


# ─────────────────────────────────────────────
# GIF 5 — Animated runtime growth chart
# ─────────────────────────────────────────────


def save_timing_gif(sizes, bf_avgs, dj_avgs, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    x = list(range(len(sizes)))
    width = 0.35

    def update(frame):
        ax.clear()
        ax.set_facecolor(PANEL)
        visible = frame + 1
        xi = x[:visible]
        bf_v = bf_avgs[:visible]
        dj_v = dj_avgs[:visible]

        bars_bf = ax.bar(
            [p - width / 2 for p in xi],
            bf_v,
            width,
            color=ACCENT2,
            alpha=0.85,
            label="Bellman-Ford O(VE)",
            zorder=3,
        )
        bars_dj = ax.bar(
            [p + width / 2 for p in xi],
            dj_v,
            width,
            color=ACCENT,
            alpha=0.85,
            label="Dijkstra O((V+E)logV)",
            zorder=3,
        )
        max_h = max(bf_avgs) * 1.15
        for bar in bars_bf:
            h = bar.get_height()
            if h > max_h * 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max_h * 0.01,
                    f"{h:.4f}s",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=WHITE,
                )
        for bar in bars_dj:
            h = bar.get_height()
            if h > max_h * 0.005:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max_h * 0.01,
                    f"{h:.4f}s",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=WHITE,
                )

        ax.set_xticks(xi)
        ax.set_xticklabels([str(sizes[i]) for i in range(visible)], color=WHITE)
        ax.set_ylim(0, max_h)
        ax.set_xlabel("Graph Size (nodes)", color=WHITE, labelpad=6)
        ax.set_ylabel("Avg Runtime (seconds)", color=WHITE, labelpad=6)
        ax.set_title(
            "Runtime Scaling: Bellman-Ford O(VE)  vs  Dijkstra O((V+E) log V)",
            color=WHITE,
            pad=10,
        )
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
        ax.grid(axis="y", color=GREY, alpha=0.3, zorder=0)
        ax.spines[:].set_color(GREY)

        if visible > 1:
            ratio = bf_avgs[visible - 1] / max(dj_avgs[visible - 1], 1e-9)
            ax.text(
                0.98,
                0.96,
                f"Speedup at n={sizes[visible - 1]}: {ratio:.0f}x",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color=YELLOW,
                bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=YELLOW, alpha=0.9),
            )

    anim = FuncAnimation(fig, update, frames=len(sizes), interval=1200)
    anim.save(filename, writer=PillowWriter(fps=1))
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# PNG 6 — Complexity comparison table
# ─────────────────────────────────────────────


def save_complexity_table(filename="complexity_table.png"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    headers = ["Property", "Bellman-Ford", "Dijkstra's"]
    rows = [
        ["Approach", "Dynamic Programming", "Greedy"],
        ["Time Complexity", "O(V · E)", "O((V+E) log V)"],
        ["Space Complexity", "O(V)", "O(V + E)"],
        ["Handles negative weights", "✓  Yes", "✗  No"],
        ["Handles negative cycles", "✓  Detects them", "✗  No"],
        ["Works distributed", "✓  Yes", "✗  No"],
        ["Practical speed", "Slow (quadratic-ish)", "Fast (log-linear)"],
        ["Best used when", "Negative weights present", "Non-negative weights only"],
    ]

    col_widths = [0.30, 0.35, 0.35]
    col_x = [0.01, 0.31, 0.66]
    row_h = 0.10
    header_y = 0.93

    for i, (h, x, w) in enumerate(zip(headers, col_x, col_widths)):
        color = ACCENT2 if i == 1 else (ACCENT if i == 2 else WHITE)
        ax.text(
            x + w / 2,
            header_y,
            h,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
        )

    # Divider line — use a thin Rectangle (axhline conflicts with axis("off"))
    divider = plt.Rectangle(
        (0.0, 0.835), 1.0, 0.005, transform=ax.transAxes, color=GREY, zorder=2
    )
    ax.add_patch(divider)

    for r, row in enumerate(rows):
        y = header_y - row_h * (r + 1) - 0.02
        bg_color = PANEL if r % 2 == 0 else BG
        rect = plt.Rectangle(
            (0.0, y - 0.005),
            1.0,
            row_h,
            transform=ax.transAxes,
            color=bg_color,
            zorder=0,
        )
        ax.add_patch(rect)
        for i, (cell, x, w) in enumerate(zip(row, col_x, col_widths)):
            ha = "left" if i == 0 else "center"
            col = WHITE
            if i > 0:
                if "✓" in cell:
                    col = GREEN
                elif "✗" in cell:
                    col = ACCENT2
                elif i == 1:
                    col = "#ffb3ae"
                else:
                    col = "#b3e5fc"
            ax.text(
                x + (0 if ha == "left" else w / 2),
                y + row_h / 2,
                cell,
                transform=ax.transAxes,
                fontsize=9,
                color=col,
                ha=ha,
                va="center",
            )

    ax.set_title(
        "Algorithm Comparison: Theory & Properties",
        color=WHITE,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# PNG 7 — Convergence curves
# ─────────────────────────────────────────────


def save_convergence_curves(
    G, target, bf_steps, bf_iters, dj_steps, dj_iters, filename="convergence_curves.png"
):
    """
    Plots how each algorithm's estimate of dist[target] decreases over steps.
    Dijkstra drops to the final value in one sharp step (greedy lock-in).
    Bellman-Ford may refine target distance gradually across multiple passes.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    def fill_series(steps):
        """Return dist[target] at each step, carrying forward the last known value."""
        filled = []
        last = None
        for s in steps:
            v = s[target] if s[target] < float("inf") else None
            if v is not None:
                last = v
            filled.append(last)
        return filled

    bf_filled = fill_series(bf_steps)
    dj_filled = fill_series(dj_steps)

    bf_x = list(range(1, len(bf_filled) + 1))
    dj_x = list(range(1, len(dj_filled) + 1))

    ax.plot(bf_x, bf_filled, color=ACCENT2, linewidth=2, label="Bellman-Ford", zorder=3)
    ax.plot(dj_x, dj_filled, color=ACCENT, linewidth=2, label="Dijkstra's", zorder=3)

    # Mark the step where each algorithm first reaches its final answer
    for series, x_vals, color, label in [
        (bf_filled, bf_x, ACCENT2, "BF converges"),
        (dj_filled, dj_x, ACCENT, "DJ converges"),
    ]:
        if not series or series[-1] is None:
            continue
        final_val = series[-1]
        for i, v in enumerate(series):
            if v == final_val:
                ax.axvline(
                    x=x_vals[i], color=color, linestyle="--", alpha=0.5, linewidth=1
                )
                ax.text(
                    x_vals[i] + len(x_vals) * 0.01,
                    final_val,
                    f"{label}\n(step {x_vals[i]})",
                    fontsize=7,
                    color=color,
                    va="bottom",
                )
                break

    ax.set_xlabel("Algorithm Step", color=WHITE, labelpad=6)
    ax.set_ylabel(f"Estimated distance to node {target}", color=WHITE, labelpad=6)
    ax.set_title(
        f"Convergence on Target Node {target}:\n"
        "Dijkstra locks in the answer in one step; Bellman-Ford refines over passes",
        color=WHITE,
        pad=10,
    )
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
    ax.grid(color=GREY, alpha=0.25, zorder=0)
    ax.spines[:].set_color(GREY)

    fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# PNG 8 — Memory usage chart
# ─────────────────────────────────────────────


def save_memory_chart(sizes, bf_mem_kb, dj_mem_kb, filename="memory_usage.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    x = list(range(len(sizes)))
    width = 0.35
    bars_bf = ax.bar(
        [p - width / 2 for p in x],
        bf_mem_kb,
        width,
        color=ACCENT2,
        alpha=0.85,
        label="Bellman-Ford",
        zorder=3,
    )
    bars_dj = ax.bar(
        [p + width / 2 for p in x],
        dj_mem_kb,
        width,
        color=ACCENT,
        alpha=0.85,
        label="Dijkstra's",
        zorder=3,
    )

    for bar in list(bars_bf) + list(bars_dj):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(bf_mem_kb) * 0.01,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=WHITE,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], color=WHITE)
    ax.set_xlabel("Graph Size (nodes)", color=WHITE, labelpad=6)
    ax.set_ylabel("Peak Memory Usage (KB)", color=WHITE, labelpad=6)
    ax.set_title("Peak Memory Usage: Bellman-Ford vs Dijkstra's", color=WHITE, pad=10)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
    ax.grid(axis="y", color=GREY, alpha=0.3, zorder=0)
    ax.spines[:].set_color(GREY)

    fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# PNG 9 — Log-scale runtime line chart
# ─────────────────────────────────────────────


def save_logscale_chart(sizes, bf_avgs, dj_avgs, filename="runtime_logscale.png"):
    """
    Line chart with a log-scale Y axis. Makes the algorithmic gap far more
    readable than a bar chart when values span several orders of magnitude
    (e.g. 0.0001s vs 50s). Also overlays theoretical O(VE) and O(VlogV)
    reference curves so the viewer can see how well the measurements fit theory.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    ax.plot(
        sizes,
        bf_avgs,
        color=ACCENT2,
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Bellman-Ford (measured)",
        zorder=3,
    )
    ax.plot(
        sizes,
        dj_avgs,
        color=ACCENT,
        linewidth=2.5,
        marker="s",
        markersize=6,
        label="Dijkstra's (measured)",
        zorder=3,
    )

    # Theoretical reference curves — scaled to match measured values at n=100
    ref_n = sizes[2] if len(sizes) > 2 else sizes[-1]  # n=100 anchor point

    # BF theoretical: O(V*E) ≈ O(V^2) with edge_factor=1
    bf_ref_scale = bf_avgs[2] / (ref_n**2) if len(bf_avgs) > 2 else 1e-7
    bf_theory = [bf_ref_scale * (n**2) for n in sizes]

    # DJ theoretical: O((V+E)logV) ≈ O(V*logV) with edge_factor=1
    dj_ref_scale = dj_avgs[2] / (ref_n * math.log2(ref_n)) if len(dj_avgs) > 2 else 1e-9
    dj_theory = [dj_ref_scale * (n * math.log2(n)) for n in sizes]

    # Reference curve labels explain the E≈V substitution used for this experiment
    ax.plot(
        sizes,
        bf_theory,
        color=ACCENT2,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
        label="O(VE) ref  [E≈V here => O(V²)]",
    )
    ax.plot(
        sizes,
        dj_theory,
        color=ACCENT,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
        label="O((V+E)logV) ref  [E≈V here => O(V logV)]",
    )

    ax.set_yscale("log")
    ax.set_xscale("log")

    # Annotate speedup at each measured point
    for n, bf, dj in zip(sizes, bf_avgs, dj_avgs):
        ratio = bf / max(dj, 1e-9)
        ax.annotate(
            f"{ratio:.0f}x",
            xy=(n, (bf * dj) ** 0.5),  # geometric mean — sits between lines
            fontsize=7,
            color=YELLOW,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.7),
        )

    ax.set_xlabel("Graph Size (nodes, log scale)", color=WHITE, labelpad=6)
    ax.set_ylabel("Avg Runtime — seconds (log scale)", color=WHITE, labelpad=6)
    ax.set_title(
        "Runtime Scaling (Log-Log): Measured vs Theoretical\n"
        "True complexities: Bellman-Ford O(VE),  Dijkstra O((V+E) log V)\n"
        "Reference curves use E \u2248 V substitution (edge_factor=1 in this experiment)",
        color=WHITE,
        pad=10,
    )
    # Push legend inside right side so it doesn't clip on the left
    ax.legend(
        fontsize=8, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, loc="lower right"
    )
    ax.grid(which="both", color=GREY, alpha=0.2, zorder=0)
    ax.spines[:].set_color(GREY)
    ax.tick_params(colors=WHITE)

    fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def run_demo():
    random.seed(42)

    graph_sizes = [10, 50, 100, 200, 500, 1000, 2000, 3000]
    num_runs = 5
    bf_avgs, dj_avgs = [], []
    bf_mems, dj_mems = [], []

    # ── 1. 10-node GIFs ──────────────────────────────────────────────────────
    print("\n=== [1/8] Generating 10-node GIFs ===")
    gif_graph = generate_graph(10, edge_factor=1)
    gif_pos = grid_layout(gif_graph)
    start, target = 0, 9

    (bf_dist, bf_prev, bf_steps, bf_currents, bf_iters, bf_visit_order) = bellman_ford(
        gif_graph, start, target, record_steps=True
    )

    (
        dj_dist,
        dj_prev,
        dj_steps,
        dj_currents,
        dj_iters,
        dj_visit_order,
        dj_settled_sets,
    ) = dijkstra(gif_graph, start, target, record_steps=True)

    save_gif_bf(
        gif_graph,
        gif_pos,
        bf_steps,
        bf_currents,
        bf_iters,
        bf_dist,
        bf_prev,
        bf_visit_order,
        "bf_10.gif",
        start,
        target,
    )

    save_gif_dj(
        gif_graph,
        gif_pos,
        dj_steps,
        dj_currents,
        dj_iters,
        dj_settled_sets,
        dj_dist,
        dj_prev,
        dj_visit_order,
        "dj_10.gif",
        start,
        target,
    )

    # ── 2. Side-by-side comparison GIF ───────────────────────────────────────
    print("\n=== [2/8] Side-by-side comparison GIF ===")
    save_comparison_gif(
        gif_graph,
        gif_pos,
        bf_steps,
        bf_currents,
        bf_iters,
        bf_dist,
        dj_steps,
        dj_currents,
        dj_iters,
        dj_settled_sets,
        "comparison_10.gif",
        start,
        target,
    )

    # ── 3. Negative-weight correctness demo ──────────────────────────────────
    print("\n=== [3/8] Negative weight correctness test ===")
    negative_weight_demo()

    # ── 4. Timing + memory runs ───────────────────────────────────────────────
    print("\n=== [4/8] Timing & memory runs ===")
    print(
        f"\n{'Size':>6}  {'BF Time':>10}  {'DJ Time':>10}  {'Speedup':>9}"
        f"  {'BF Mem KB':>10}  {'DJ Mem KB':>10}"
    )
    print("-" * 64)

    for size in graph_sizes:
        bf_times, dj_times = [], []
        bf_mem_runs, dj_mem_runs = [], []

        for _ in range(num_runs):
            G = generate_graph(size, edge_factor=1)
            tracemalloc.start()
            t0 = time.time()
            bellman_ford(G, 0, size - 1)
            bf_times.append(time.time() - t0)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            bf_mem_runs.append(peak / 1024)

        for _ in range(num_runs):
            G = generate_graph(size, edge_factor=1)
            tracemalloc.start()
            t0 = time.time()
            dijkstra(G, 0, size - 1)
            dj_times.append(time.time() - t0)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            dj_mem_runs.append(peak / 1024)

        bf_avg = sum(bf_times) / num_runs
        dj_avg = sum(dj_times) / num_runs
        bf_mem = sum(bf_mem_runs) / num_runs
        dj_mem = sum(dj_mem_runs) / num_runs
        bf_avgs.append(bf_avg)
        dj_avgs.append(dj_avg)
        bf_mems.append(bf_mem)
        dj_mems.append(dj_mem)

        ratio = bf_avg / max(dj_avg, 1e-9)
        print(
            f"{size:>6}  {bf_avg:>10.4f}  {dj_avg:>10.4f}  {ratio:>8.0f}x"
            f"  {bf_mem:>10.1f}  {dj_mem:>10.1f}"
        )

    # ── 5. Animated timing chart ──────────────────────────────────────────────
    print("\n=== [5/9] Animated timing chart ===")
    save_timing_gif(graph_sizes, bf_avgs, dj_avgs, "timing_growth.gif")

    # ── 6. Complexity table ───────────────────────────────────────────────────
    print("\n=== [6/9] Complexity comparison table ===")
    save_complexity_table("complexity_table.png")

    # ── 7. Convergence curves ─────────────────────────────────────────────────
    print("\n=== [7/9] Convergence curves ===")
    save_convergence_curves(
        gif_graph,
        target,
        bf_steps,
        bf_iters,
        dj_steps,
        dj_iters,
        "convergence_curves.png",
    )

    # ── 8. Memory usage chart ─────────────────────────────────────────────────
    print("\n=== [8/9] Memory usage chart ===")
    save_memory_chart(graph_sizes, bf_mems, dj_mems, "memory_usage.png")

    # ── 9. Log-scale runtime line chart ──────────────────────────────────────
    print("\n=== [9/9] Log-scale runtime line chart ===")
    save_logscale_chart(graph_sizes, bf_avgs, dj_avgs, "runtime_logscale.png")

    print("\n✓ All outputs ready:")
    print(
        "  bf_10.gif                  — Bellman-Ford (settled=green, path=gold, visit order on final frame)"
    )
    print(
        "  dj_10.gif                  — Dijkstra (settled=green, path=gold, visit order on final frame)"
    )
    print("  comparison_10.gif          — Side-by-side on identical graph")
    print(
        "  negative_weight_comparison.png — Dijkstra provably wrong on neg-weight graph"
    )
    print("  timing_growth.gif          — Animated runtime scaling bar chart")
    print("  complexity_table.png       — Theory & properties comparison table")
    print(
        "  convergence_curves.png     — Distance-to-target over steps, both algorithms"
    )
    print("  memory_usage.png           — Peak memory per graph size, both algorithms")
    print(
        "  runtime_logscale.png       — Log-log line chart with theoretical O() reference curves"
    )


if __name__ == "__main__":
    run_demo()
