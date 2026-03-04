"""
generates a planar city graph using voronoi diagrams
saves as graphml with correct .graphml extension
"""

import numpy as np
import networkx as nx
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import os


def generate_city_graph(num_nodes=19, seed=None):
    if seed is not None:
        np.random.seed(seed)

    points = np.random.rand(num_nodes, 2)
    vor    = Voronoi(points)
    G      = nx.Graph()

    for i, point in enumerate(points):
        node_name = f"({point[0]:.6f}, {point[1]:.6f})"
        G.add_node(node_name, x=float(point[0]), y=float(point[1]))

    for p1_idx, p2_idx in vor.ridge_points:
        p1    = points[p1_idx]
        p2    = points[p2_idx]
        node1 = f"({p1[0]:.6f}, {p1[1]:.6f})"
        node2 = f"({p2[0]:.6f}, {p2[1]:.6f})"
        w     = float(np.linalg.norm(p1 - p2))
        G.add_edge(node1, node2, weight=w)

    # make sure the graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            comp1, comp2 = list(components[i]), list(components[i+1])
            best_pair = None
            min_dist  = float("inf")
            for n1 in comp1:
                pos1 = np.array([G.nodes[n1]["x"], G.nodes[n1]["y"]])
                for n2 in comp2:
                    pos2 = np.array([G.nodes[n2]["x"], G.nodes[n2]["y"]])
                    d = np.linalg.norm(pos1 - pos2)
                    if d < min_dist:
                        min_dist  = d
                        best_pair = (n1, n2)
            if best_pair:
                G.add_edge(best_pair[0], best_pair[1], weight=float(min_dist))

    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def save_graph(G, filename="g1.graphml"):
    # FIX 4: filename is now .graphml to match the actual GraphML format
    nx.write_graphml(G, filename)
    print(f"graph saved -> {filename}")


def load_graph(filename="g1.graphml"):
    # FIX 4: updated default filename
    G = nx.read_graphml(filename)
    print(f"graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def visualize_graph(G, filename="g1_visualization.png", title="city graph"):
    os.makedirs("graph_photos", exist_ok=True)
    save_path = os.path.join("graph_photos", filename)

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(10, 8))

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#adb5bd", lw=1.4, alpha=0.7, zorder=1)

    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    ax.scatter(xs, ys, s=220, c="#4361ee", alpha=0.85,
               edgecolors="#023e8a", linewidths=1.8, zorder=2)

    if len(G.nodes()) <= 30:
        for i, node in enumerate(G.nodes()):
            ax.text(pos[node][0], pos[node][1], str(i),
                    fontsize=9, ha="center", va="center",
                    color="white", weight="bold", zorder=3)

    ax.set_title(f"{title} - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                 fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"visualization saved -> {save_path}")
    plt.close()


def analyze_graph(G):
    print("\ngraph stats:")
    print(f"  nodes      : {G.number_of_nodes()}")
    print(f"  edges      : {G.number_of_edges()}")
    print(f"  connected  : {nx.is_connected(G)}")
    print(f"  avg degree : {np.mean([d for _, d in G.degree()]):.2f}")
    if nx.is_connected(G):
        print(f"  avg path   : {nx.average_shortest_path_length(G):.2f}")
        print(f"  diameter   : {nx.diameter(G)}")
    print(f"  clustering : {nx.average_clustering(G):.3f}")
    # FIX 1: show degree distribution so you know the real action space per node
    degrees = sorted([d for _, d in G.degree()])
    print(f"  min degree : {min(degrees)}  (min valid actions = {min(degrees)+1})")
    print(f"  max degree : {max(degrees)}  (max valid actions = {max(degrees)+1})")


if __name__ == "__main__":
    print("generating city graph (19 nodes, seed=42)")
    G = generate_city_graph(num_nodes=19, seed=42)
    analyze_graph(G)
    # FIX 4: save with correct .graphml extension
    save_graph(G, "g1.graphml")
    visualize_graph(G, "g1_visualization.png", "city planar graph")
    print("\ndone")
    print("  g1.graphml                        - graph file (GraphML format)")
    print("  graph_photos/g1_visualization.png - visualization")
