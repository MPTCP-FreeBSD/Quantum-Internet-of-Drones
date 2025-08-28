import itertools
import networkx as nx


def get_tp_ct_st():

    # -----------------------------
    # Parameters
    # -----------------------------
    N = 5  # number of drones/nodes
    kappa = 1.0   # per-link crosstalk coefficient
    gamma = 1.0   # per-node crosstalk coefficient
    MAX_REPEATERS = 2  # "at most 2 repeaters" rule

    # -----------------------------
    # Helper functions
    # -----------------------------
    def build_topology(name):
        """Return a NetworkX Graph for given topology name."""
        G = nx.Graph()
        G.add_nodes_from(range(1, N + 1))
        if name == "Bus":
            edges = [(i, i + 1) for i in range(1, N)]
        elif name == "Ring":
            edges = [(i, i + 1) for i in range(1, N)] + [(N, 1)]
        elif name == "Star":
            hub = 1
            edges = [(hub, i) for i in range(2, N + 1)]
        elif name == "Mesh":
            edges = list(itertools.combinations(range(1, N + 1), 2))
        else:
            raise ValueError(f"Unknown topology {name}")
        G.add_edges_from(edges)
        return G

    def count_crosstalk(G):
        """Compute normalized crosstalk_strength for all-pairs sessions."""
        sessions = list(itertools.combinations(G.nodes, 2))
        edge_loads = {e: 0 for e in G.edges}
        node_loads = {n: 0 for n in G.nodes}

        for u, v in sessions:
            path = nx.shortest_path(G, u, v)
            repeaters = path[1:-1]
            if len(repeaters) > MAX_REPEATERS:
                repeaters = repeaters[:MAX_REPEATERS]
            # Count edge usage
            for e in zip(path, path[1:]):
                e_sorted = tuple(sorted(e))
                edge_loads[e_sorted] += 1
            # Count node (repeater) usage
            for n in repeaters:
                node_loads[n] += 1

        # Raw crosstalk
        edge_term = sum(kappa * (L * (L - 1) / 2) for L in edge_loads.values())
        node_term = sum(gamma * (B * (B - 1) / 2) for B in node_loads.values())
        raw_crosstalk = edge_term + node_term
        raw_crosstalk = edge_term 

        # Max possible crosstalk (if all sessions go through same link + repeaters)
        max_edge_load = len(sessions)  # all pairs on same link
        max_node_load = len(sessions)  # all pairs through same repeater
        max_crosstalk = kappa * (max_edge_load * (max_edge_load - 1) / 2) \
                        + gamma * (max_node_load * (max_node_load - 1) / 2)

        # Normalize between 0 and 1
        norm_crosstalk = raw_crosstalk / max_crosstalk if max_crosstalk > 0 else 0
        return norm_crosstalk, edge_loads, node_loads

    # -----------------------------
    # Run for all topologies
    # -----------------------------
    topologies = ["Bus", "Ring", "Star", "Mesh"]
    results = {}

    for topo in topologies:
        G = build_topology(topo)
        norm_crosstalk, edge_loads, node_loads = count_crosstalk(G)
        results[topo] = norm_crosstalk
        print(f"=== {topo} ===")
        print("Edges:", list(G.edges))
        print("Edge loads:", edge_loads)
        print("Node loads:", node_loads)
        print(f"Crosstalk Strength (0–1) = {norm_crosstalk:.3f}")
        print()

    # -----------------------------
    # Summary ranking
    # -----------------------------
    print("=== Crosstalk Strength Ranking (0–1) ===")
    for topo, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"{topo}: {score:.3f}")
    
    return results
