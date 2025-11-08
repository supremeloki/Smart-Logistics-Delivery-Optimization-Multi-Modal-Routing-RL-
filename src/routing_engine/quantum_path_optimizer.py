import random
import yaml
import os
import logging
import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumPathOptimizer:
    def __init__(
        self,
        config_path="conf/environments/prod.yaml",
        environment="dev",
        graph_path="data_nexus/road_network_graph/preprocessed_tehran_graph.gml",
    ):
        self.config = self._load_config(config_path)
        self.qpo_config = self.config["environments"][environment][
            "quantum_path_optimizer"
        ]
        self.graph = self._load_or_generate_graph(graph_path)
        self.nodes = list(self.graph.nodes())
        self.adjacency_matrix = self._build_adjacency_matrix()

        logger.info("QuantumPathOptimizer initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    quantum_path_optimizer:
      enabled: true
      num_qubits: 10 # Max nodes to consider for quantum-like path
      num_reads: 100 # Iterations for simulated annealing
      temperature: 100.0
      cooldown_factor: 0.95
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_or_generate_graph(self, graph_path):
        if os.path.exists(graph_path):
            return nx.read_gml(graph_path)
        else:
            logger.warning(
                f"Graph file not found at {graph_path}. Generating a small dummy graph."
            )
            G_temp = nx.MultiDiGraph()
            for i in range(15):
                G_temp.add_node(i, x=random.uniform(0, 10), y=random.uniform(0, 10))
            for i in range(20):
                u, v = random.randint(0, 14), random.randint(0, 14)
                if u != v:
                    G_temp.add_edge(
                        u,
                        v,
                        key=0,
                        travel_time=random.uniform(10, 100),
                        length=random.uniform(100, 500),
                    )
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            nx.write_gml(G_temp, graph_path)
            return G_temp

    def _build_adjacency_matrix(self):
        node_map = {node: i for i, node in enumerate(self.nodes)}
        n = len(self.nodes)
        adj_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(adj_matrix, 0)
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            u_idx, v_idx = node_map[u], node_map[v]
            adj_matrix[u_idx, v_idx] = min(
                adj_matrix[u_idx, v_idx], data.get("travel_time", np.inf)
            )
        return adj_matrix

    def _calculate_path_cost(self, path_indices):
        cost = 0
        for i in range(len(path_indices) - 1):
            u_idx, v_idx = path_indices[i], path_indices[i + 1]
            edge_cost = self.adjacency_matrix[u_idx, v_idx]
            if edge_cost == np.inf:
                return np.inf  # Invalid path
            cost += edge_cost
        return cost

    def _simulate_quantum_annealing(self, start_idx, end_idx):
        n = len(self.nodes)
        num_qubits = self.qpo_config["num_qubits"]
        num_reads = self.qpo_config["num_reads"]
        temperature = self.qpo_config["temperature"]
        cooldown_factor = self.qpo_config["cooldown_factor"]

        if n > num_qubits:
            # Simplify graph for quantum-like annealing if too many nodes
            # For a creative sandbox, we'll just pick N closest nodes.
            relevant_nodes = [start_idx, end_idx]

            distances = []
            for i in range(n):
                if i != start_idx and i != end_idx:
                    dist_to_start = (
                        self.adjacency_matrix[start_idx, i]
                        if self.adjacency_matrix[start_idx, i] != np.inf
                        else self.adjacency_matrix[i, start_idx]
                    )
                    dist_to_end = (
                        self.adjacency_matrix[i, end_idx]
                        if self.adjacency_matrix[i, end_idx] != np.inf
                        else self.adjacency_matrix[end_idx, i]
                    )
                    distances.append(
                        (i, min(dist_to_start, dist_to_end))
                    )  # Simplified distance metric
            distances.sort(key=lambda x: x[1])

            for node_idx, _ in distances:
                if len(relevant_nodes) < num_qubits:
                    relevant_nodes.append(node_idx)
                else:
                    break

            # Re-map indices for the smaller subset
            original_to_subset_map = {node: i for i, node in enumerate(relevant_nodes)}
            subset_to_original_map = {i: node for i, node in enumerate(relevant_nodes)}

            current_start_idx = original_to_subset_map.get(start_idx, 0)
            current_end_idx = original_to_subset_map.get(end_idx, 0)

            # Rebuild a temporary adjacency matrix for the subset
            subset_adj_matrix = np.full(
                (len(relevant_nodes), len(relevant_nodes)), np.inf
            )
            for i_orig in relevant_nodes:
                for j_orig in relevant_nodes:
                    i_subset = original_to_subset_map[i_orig]
                    j_subset = original_to_subset_map[j_orig]
                    subset_adj_matrix[i_subset, j_subset] = self.adjacency_matrix[
                        i_orig, j_orig
                    ]
            current_adj_matrix = subset_adj_matrix
        else:
            current_start_idx = start_idx
            current_end_idx = end_idx
            current_adj_matrix = self.adjacency_matrix
            subset_to_original_map = {i: node for i, node in enumerate(self.nodes)}

        best_path_indices = []
        best_cost = np.inf

        for _ in range(num_reads):
            current_path = [current_start_idx]
            current_cost = 0
            visited = {current_start_idx}

            while (
                current_path[-1] != current_end_idx and len(current_path) < num_qubits
            ):
                last_node_idx = current_path[-1]

                # Identify available next steps (neighbors not yet visited)
                next_possible_nodes = []
                for neighbor_idx in range(len(current_adj_matrix)):
                    if (
                        current_adj_matrix[last_node_idx, neighbor_idx] != np.inf
                        and neighbor_idx not in visited
                    ):
                        next_possible_nodes.append(neighbor_idx)

                if not next_possible_nodes:
                    current_cost = np.inf  # Dead end
                    break

                # Quantum-inspired choice: sample based on inverse cost and temperature
                energies = [
                    current_adj_matrix[last_node_idx, n_idx]
                    for n_idx in next_possible_nodes
                ]
                # Invert energies so lower cost means higher "probability"
                inverse_energies = [
                    1.0 / (e + 1e-9) for e in energies
                ]  # Add small epsilon to avoid div by zero
                probabilities = np.exp(np.array(inverse_energies) / temperature)
                probabilities /= probabilities.sum()

                next_node_idx = np.random.choice(next_possible_nodes, p=probabilities)

                current_path.append(next_node_idx)
                visited.add(next_node_idx)
                current_cost += current_adj_matrix[last_node_idx, next_node_idx]

            if current_path[-1] == current_end_idx and current_cost < best_cost:
                best_cost = current_cost
                best_path_indices = current_path

            temperature *= cooldown_factor  # Simulate annealing

        # Map back to original node IDs
        original_best_path = [subset_to_original_map[idx] for idx in best_path_indices]
        return {"path": original_best_path, "cost": best_cost}

    def find_optimized_path(self, start_node_id: int, end_node_id: int):
        if not self.qpo_config["enabled"]:
            logger.info("Quantum Path Optimizer is disabled. Using A* or fallback.")
            return None  # Fallback to a classical router

        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            logger.error(
                f"Start node {start_node_id} or end node {end_node_id} not in graph."
            )
            return None

        start_idx = self.nodes.index(start_node_id)
        end_idx = self.nodes.index(end_node_id)

        logger.info(
            f"Finding quantum-inspired path from {start_node_id} to {end_node_id}..."
        )
        result = self._simulate_quantum_annealing(start_idx, end_idx)

        if result["cost"] == np.inf:
            logger.warning(
                f"No valid quantum-inspired path found from {start_node_id} to {end_node_id}."
            )
            return None

        logger.info(
            f"Quantum-inspired path found: {result['path']} with cost {result['cost']:.2f}"
        )
        return result


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    graph_file = "data_nexus/road_network_graph/quantum_test_graph.gml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)

    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    quantum_path_optimizer:
      enabled: true
      num_qubits: 8
      num_reads: 50
      temperature: 50.0
      cooldown_factor: 0.9
"""
            )

    # Create a test graph for demonstration
    G_test = nx.MultiDiGraph()
    G_test.add_edges_from(
        [
            (0, 1, {"travel_time": 10}),
            (0, 2, {"travel_time": 50}),
            (1, 3, {"travel_time": 20}),
            (1, 4, {"travel_time": 70}),
            (2, 4, {"travel_time": 15}),
            (2, 5, {"travel_time": 80}),
            (3, 6, {"travel_time": 30}),
            (4, 6, {"travel_time": 10}),
            (5, 6, {"travel_time": 25}),
        ],
        key=0,
    )
    nx.write_gml(G_test, graph_file)
    print("Test graph generated.")

    q_optimizer = QuantumPathOptimizer(config_file, graph_path=graph_file)

    # Find path
    start_node = 0
    end_node = 6
    path_result = q_optimizer.find_optimized_path(start_node, end_node)

    if path_result:
        print(f"\nOptimal path from {start_node} to {end_node}: {path_result['path']}")
        print(f"Total simulated quantum cost: {path_result['cost']:.2f}")
    else:
        print(
            f"\nCould not find a path from {start_node} to {end_node} using quantum-inspired optimization."
        )

    # Test with a larger graph (will trigger subset selection)
    G_large = nx.MultiDiGraph()
    for i in range(20):
        G_large.add_node(i, x=random.uniform(0, 10), y=random.uniform(0, 10))
    for i in range(40):
        u, v = random.randint(0, 19), random.randint(0, 19)
        if u != v:
            G_large.add_edge(
                u,
                v,
                key=0,
                travel_time=random.uniform(10, 100),
                length=random.uniform(100, 500),
            )
    nx.write_gml(G_large, "data_nexus/road_network_graph/large_quantum_test_graph.gml")
    print("\nLarge test graph generated.")

    q_optimizer_large = QuantumPathOptimizer(
        config_file,
        graph_path="data_nexus/road_network_graph/large_quantum_test_graph.gml",
    )
    start_node_large = random.choice(list(G_large.nodes()))
    end_node_large = random.choice(list(G_large.nodes()))
    while end_node_large == start_node_large:
        end_node_large = random.choice(list(G_large.nodes()))

    path_result_large = q_optimizer_large.find_optimized_path(
        start_node_large, end_node_large
    )
    if path_result_large:
        print(
            f"\nOptimal path from {start_node_large} to {end_node_large} (large graph): {path_result_large['path']}"
        )
        print(f"Total simulated quantum cost: {path_result_large['cost']:.2f}")
    else:
        print(
            f"\nCould not find a path from {start_node_large} to {end_node_large} using quantum-inspired optimization (large graph)."
        )

    print("\nQuantum Path Optimizer demo complete.")
