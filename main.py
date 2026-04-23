"""
Graph Algorithms Visualization: Bellman‑Ford vs. Dijkstra

This program generates random undirected graphs, runs Bellman‑Ford and Dijkstra,
and compares their performance. For a small graph (10 nodes), it creates animated
GIFs showing how each algorithm explores the graph step by step.

The code is structured to be readable and easy to modify for educational purposes.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import heapq
from matplotlib.animation import FuncAnimation, PillowWriter
import math
from collections import deque

# ============================================================================
# GRAPH GENERATION
# ============================================================================

def ensure_reachability(G, start, target):
    """
    Guarantees that `target` is reachable from `start` in an undirected graph.
    If not, adds a simple path (with at most 3 intermediate nodes) to connect them.
    """
    # BFS to check reachability
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

    # Build a connecting path using some free nodes
    available = list(set(G.nodes()) - {start, target})
    random.shuffle(available)
    path = [start] + available[:random.randint(1, min(3, len(available)))] + [target]
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v):
            weight = random.randint(1, 10)
            G.add_edge(u, v, weight=weight)
    return G

def generate_graph(n, edge_factor=2, max_edges=None):
    """
    Creates a random undirected graph with `n` nodes and approximately
    `n * edge_factor` edges. Ensures that node `n-1` is reachable from node 0.
    """
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    num_edges = n * edge_factor if max_edges is None else min(n * edge_factor, max_edges)
    edges_added = 0
    while edges_added < num_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and not G.has_edge(u, v):
            weight = random.randint(1, 10)
            G.add_edge(u, v, weight=weight)
            edges_added += 1

    G = ensure_reachability(G, start=0, target=n-1)
    return G

# ============================================================================
# GRAPH LAYOUT AND DRAWING
# ============================================================================

def grid_layout(G):
    """
    Arranges nodes in a grid (left‑to‑right, top‑to‑bottom) for consistent plotting.
    """
    n = len(G.nodes())
    cols = int(math.ceil(math.sqrt(n)))
    pos = {}
    for i, node in enumerate(sorted(G.nodes())):
        row = i // cols
        col = i % cols
        pos[node] = (col, -row)          # negative row to place nodes from top to bottom
    return pos

def draw_graph(G, pos, dist, current, ax, start, target):
    """
    Draws the graph with node colors:
      - yellow: currently being processed
      - green: already reached (distance < inf)
      - red: not yet reached
    """
    ax.clear()
    colors = []
    for node in G.nodes():
        if node == current:
            colors.append("yellow")
        elif dist[node] < float('inf'):
            colors.append("green")
        else:
            colors.append("red")
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=colors,
            node_size=500, font_size=8)

    # Edge labels only for small graphs (avoid clutter)
    if len(G.nodes()) <= 50:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                      ax=ax, font_size=7)

    ax.text(0, 1.1, f"Start: {start}  Target: {target}",
            transform=ax.transAxes, fontsize=10)
    ax.set_title("Graph Traversal Visualization")

# ============================================================================
# ALGORITHMS
# ============================================================================

def bellman_ford(G, start, target, record_steps=False):
    """
    Bellman‑Ford algorithm for undirected graphs.
    If record_steps=True, returns step‑by‑step distance dictionaries for animation.
    """
    dist = {node: float('inf') for node in G.nodes()}
    dist[start] = 0
    steps = [] if record_steps else None
    currents = [] if record_steps else None

    for _ in range(len(G.nodes()) - 1):
        for u, v, data in G.edges(data=True):
            w = data['weight']
            # Process edge in both directions (undirected)
            for a, b in [(u, v), (v, u)]:
                if dist[a] + w < dist[b]:
                    dist[b] = dist[a] + w
                if record_steps:
                    steps.append(dist.copy())
                    currents.append(a)      # source node of this half‑edge
    return dist, steps, currents

def dijkstra(G, start, target, record_steps=False):
    """
    Dijkstra's algorithm using a priority queue.
    If record_steps=True, returns step‑by‑step distance dictionaries for animation.
    """
    dist = {node: float('inf') for node in G.nodes()}
    dist[start] = 0
    visited = set()
    steps = [] if record_steps else None
    currents = [] if record_steps else None

    heap = [(0, start)]
    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v in G.neighbors(u):
            w = G[u][v]['weight']
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
                if record_steps:
                    steps.append(dist.copy())
                    currents.append(u)      # node whose edge caused the update
    return dist, steps, currents

# ============================================================================
# GIF CREATION
# ============================================================================

def save_gif(G, pos, steps, currents, filename, start, target):
    """
    Creates an animated GIF from the algorithm's step sequence.
    Limits the number of frames to 200 to keep the file size reasonable.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sample frames if there are too many
    if len(steps) > 200:
        indices = list(range(0, len(steps), len(steps)//200))
        steps_to_save = [steps[i] for i in indices]
        currents_to_save = [currents[i] for i in indices]
    else:
        steps_to_save = steps
        currents_to_save = currents

    def update(frame):
        draw_graph(G, pos, steps_to_save[frame],
                   currents_to_save[frame], ax, start, target)

    anim = FuncAnimation(fig, update, frames=len(steps_to_save), interval=200)
    anim.save(filename, writer=PillowWriter(fps=2))
    plt.close(fig)

# ============================================================================
# MAIN DEMO
# ============================================================================

def run_demo():
    """Main driver: creates GIFs for the 10‑node graph and times both algorithms."""
    graph_sizes = [10, 50, 100, 200, 500, 1000]   # Bellman‑Ford becomes slow beyond 1000
    num_runs = 5

    # -------- Generate the 10‑node graph once for both GIFs --------
    if 10 in graph_sizes:
        print("\n=== Generating shared 10‑node graph for GIFs ===")
        gif_graph = generate_graph(10, edge_factor=1)
        gif_pos = grid_layout(gif_graph)
        start, target = 0, 9

        print("Running Bellman‑Ford on the shared graph to create GIF...")
        bf_dist, bf_steps, bf_currents = bellman_ford(gif_graph, start, target, record_steps=True)
        if bf_dist[target] < float('inf'):
            save_gif(gif_graph, gif_pos, bf_steps, bf_currents, "bf_10.gif", start, target)
        else:
            print("Warning: target not reachable for Bellman‑Ford GIF; skipping.")

        print("Running Dijkstra on the shared graph to create GIF...")
        dj_dist, dj_steps, dj_currents = dijkstra(gif_graph, start, target, record_steps=True)
        if dj_dist[target] < float('inf'):
            save_gif(gif_graph, gif_pos, dj_steps, dj_currents, "dj_10.gif", start, target)
        else:
            print("Warning: target not reachable for Dijkstra GIF; skipping.")

    # -------- Time the algorithms on fresh graphs for each size --------
    for size in graph_sizes:
        print(f"\n===== Graph size: {size} =====")
        bf_times = []
        dj_times = []

        # Bellman‑Ford timing runs
        for run in range(num_runs):
            G = generate_graph(size, edge_factor=1)
            start, target = 0, size - 1
            start_time = time.time()
            dist, _, _ = bellman_ford(G, start, target, record_steps=False)
            end_time = time.time()
            bf_times.append(end_time - start_time)

        # Dijkstra timing runs
        for run in range(num_runs):
            G = generate_graph(size, edge_factor=1)
            start, target = 0, size - 1
            start_time = time.time()
            dist, _, _ = dijkstra(G, start, target, record_steps=False)
            end_time = time.time()
            dj_times.append(end_time - start_time)

        print(f"Average Bellman‑Ford time: {sum(bf_times)/num_runs:.4f} s")
        print(f"Average Dijkstra time:     {sum(dj_times)/num_runs:.4f} s")

if __name__ == "__main__":
    run_demo()
