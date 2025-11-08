import random
import yaml
import os
import logging
from collections import deque
import numpy as np
import networkx as nx
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumOptimizationSimulator:
    def __init__(
        self,
        config_path="conf/environments/prod.yaml",
        environment="dev",
        graph_path="data_nexus/road_network_graph/preprocessed_tehran_graph.gml",
    ):
        self.config = self._load_config(config_path)
        self.qsim_config = self.config["environments"][environment]["quantum_simulator"]
        self.graph = self._load_or_generate_graph(graph_path)
        self.nodes = list(self.graph.nodes())
        self.node_map = {node: i for i, node in enumerate(self.nodes)}
        self.reverse_node_map = {i: node for i, node in enumerate(self.nodes)}
        self.base_adjacency_matrix = self._build_adjacency_matrix()

        self.simulated_paths = deque(
            maxlen=self.qsim_config["history_size"]
        )  # Store past simulation results
        logger.info("QuantumOptimizationSimulator initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    quantum_simulator:
      enabled: true
      max_concurrent_tasks: 5 # Max simultaneous quantum-like optimizations
      num_qubits_per_task: 12 # Effective search space size
      annealing_iterations: 200
      initial_temperature: 150.0
      cooling_rate: 0.98
      history_size: 100
      cost_weights:
        travel_time: 1.0
        load_penalty: 0.2
        driver_fatigue_penalty: 0.5
        demand_incentive: -0.3 # Negative for incentive to prefer high demand
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
            for i in range(20):
                G_temp.add_node(i, x=random.uniform(0, 10), y=random.uniform(0, 10))
            for i in range(30):
                u, v = random.randint(0, 19), random.randint(0, 19)
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
        n = len(self.nodes)
        adj_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(adj_matrix, 0)
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            u_idx, v_idx = self.node_map[u], self.node_map[v]
            adj_matrix[u_idx, v_idx] = min(
                adj_matrix[u_idx, v_idx], data.get("travel_time", np.inf)
            )  # Base travel time
        return adj_matrix

    def _calculate_dynamic_cost(self, u_idx, v_idx, current_state: Dict[str, Any]):
        original_u, original_v = (
            self.reverse_node_map[u_idx],
            self.reverse_node_map[v_idx],
        )
        edge_data = self.graph.get_edge_data(original_u, original_v, key=0)

        if edge_data is None:
            return np.inf

        travel_time = edge_data.get("travel_time", np.inf) * edge_data.get(
            "traffic_factor", 1.0
        )

        cost = travel_time * self.qsim_config["cost_weights"]["travel_time"]

        # Add penalties/incentives based on current_state
        if "driver_current_load" in current_state:
            cost += (
                current_state["driver_current_load"]
                * self.qsim_config["cost_weights"]["load_penalty"]
            )
        if "driver_fatigue_score" in current_state:
            cost += (
                current_state["driver_fatigue_score"]
                * self.qsim_config["cost_weights"]["driver_fatigue_penalty"]
            )
        if "node_demand_density" in current_state:
            # Incentive to go to high demand areas (negative cost)
            cost += (
                current_state["node_demand_density"]
                * self.qsim_config["cost_weights"]["demand_incentive"]
            )

        return cost

    def _simulate_one_quantum_annealing_task(
        self, start_node_idx, end_node_idx, current_state: Dict[str, Any]
    ):
        n_graph_nodes = len(self.nodes)
        num_qubits = self.qsim_config["num_qubits_per_task"]
        annealing_iterations = self.qsim_config["annealing_iterations"]
        current_temperature = self.qsim_config["initial_temperature"]
        cooling_rate = self.qsim_config["cooling_rate"]

        if n_graph_nodes > num_qubits:
            # Subgraph selection based on start/end node and their vicinity
            relevant_nodes_orig_ids = nx.single_source_dijkstra_path_length(
                self.graph, self.reverse_node_map[start_node_idx], cutoff=5
            ).keys()
            relevant_nodes_orig_ids.update(
                nx.single_source_dijkstra_path_length(
                    self.graph, self.reverse_node_map[end_node_idx], cutoff=5
                ).keys()
            )

            # Ensure start/end are in the subgraph
            if self.reverse_node_map[start_node_idx] not in relevant_nodes_orig_ids:
                relevant_nodes_orig_ids.add(self.reverse_node_map[start_node_idx])
            if self.reverse_node_map[end_node_idx] not in relevant_nodes_orig_ids:
                relevant_nodes_orig_ids.add(self.reverse_node_map[end_node_idx])

            # Select up to num_qubits relevant nodes, prioritizing start/end and neighbors
            relevant_nodes_orig_ids_list = list(relevant_nodes_orig_ids)
            random.shuffle(
                relevant_nodes_orig_ids_list
            )  # Randomize to pick a diverse set if too many
            if len(relevant_nodes_orig_ids_list) > num_qubits:
                relevant_nodes_orig_ids_list = relevant_nodes_orig_ids_list[:num_qubits]

            subgraph_node_map = {
                node_id: i for i, node_id in enumerate(relevant_nodes_orig_ids_list)
            }
            subgraph_reverse_node_map = {
                i: node_id for i, node_id in enumerate(relevant_nodes_orig_ids_list)
            }

            subgraph_start_idx = subgraph_node_map.get(
                self.reverse_node_map[start_node_idx]
            )
            subgraph_end_idx = subgraph_node_map.get(
                self.reverse_node_map[end_node_idx]
            )

            if subgraph_start_idx is None or subgraph_end_idx is None:
                logger.warning(
                    f"Start/end node {self.reverse_node_map[start_node_idx]}/{self.reverse_node_map[end_node_idx]} not in selected quantum subgraph. Fallback to default."
                )
                return {
                    "path": [
                        self.reverse_node_map[start_node_idx],
                        self.reverse_node_map[end_node_idx],
                    ],
                    "cost": np.inf,
                }

            # Build dynamic adjacency matrix for the subgraph
            dynamic_adj_matrix = np.full(
                (len(subgraph_node_map), len(subgraph_node_map)), np.inf
            )
            for i_orig in relevant_nodes_orig_ids_list:
                for j_orig in relevant_nodes_orig_ids_list:
                    if i_orig == j_orig:
                        dynamic_adj_matrix[
                            subgraph_node_map[i_orig], subgraph_node_map[j_orig]
                        ] = 0
                    else:
                        cost = self._calculate_dynamic_cost(
                            self.node_map[i_orig], self.node_map[j_orig], current_state
                        )
                        dynamic_adj_matrix[
                            subgraph_node_map[i_orig], subgraph_node_map[j_orig]
                        ] = cost

            current_adj_matrix = dynamic_adj_matrix
            current_reverse_node_map = subgraph_reverse_node_map
            start_idx_local = subgraph_start_idx
            end_idx_local = subgraph_end_idx

        else:  # Full graph if small enough
            current_adj_matrix = np.full(self.base_adjacency_matrix.shape, np.inf)
            for i in range(n_graph_nodes):
                for j in range(n_graph_nodes):
                    current_adj_matrix[i, j] = self._calculate_dynamic_cost(
                        i, j, current_state
                    )

            self.node_map
            current_reverse_node_map = self.reverse_node_map
            start_idx_local = start_node_idx
            end_idx_local = end_node_idx

        best_path_indices = []
        best_cost = np.inf

        for _ in range(annealing_iterations):
            path = [start_idx_local]
            cost = 0
            visited = {start_idx_local}

            while (
                path[-1] != end_idx_local and len(path) < num_qubits
            ):  # Max path length limit
                last_node_in_path = path[-1]

                # Potential next nodes (neighbors not yet visited)
                possible_next_nodes = []
                for neighbor_idx in range(len(current_adj_matrix)):
                    if (
                        current_adj_matrix[last_node_in_path, neighbor_idx] != np.inf
                        and neighbor_idx not in visited
                    ):
                        possible_next_nodes.append(neighbor_idx)

                if not possible_next_nodes:
                    cost = np.inf  # Dead end
                    break

                # Quantum-inspired probabilistic choice based on "energy" (cost)
                energies = [
                    current_adj_matrix[last_node_in_path, next_n_idx]
                    for next_n_idx in possible_next_nodes
                ]

                # Boltzmann distribution for probabilities
                probabilities = np.exp(-np.array(energies) / current_temperature)
                probabilities /= probabilities.sum()

                chosen_next_node = np.random.choice(
                    possible_next_nodes, p=probabilities
                )

                path.append(chosen_next_node)
                visited.add(chosen_next_node)
                cost += current_adj_matrix[last_node_in_path, chosen_next_node]

            if path[-1] == end_idx_local and cost < best_cost:
                best_cost = cost
                best_path_indices = path

            current_temperature *= cooling_rate  # Annealing schedule

        original_best_path = (
            [current_reverse_node_map[idx] for idx in best_path_indices]
            if best_path_indices
            else []
        )
        return {"path": original_best_path, "cost": best_cost}

    def simulate_optimization(self, tasks: List[Dict[str, Any]]):
        """
        Simulates multiple concurrent quantum-inspired optimization tasks.
        Each task is a dict: {'task_id': str, 'start_node': int, 'end_node': int, 'current_state': dict}
        """
        if not self.qsim_config["enabled"]:
            logger.info("Quantum Optimization Simulator is disabled.")
            return []

        results = []
        for task in tasks[: self.qsim_config["max_concurrent_tasks"]]:
            task_id = task["task_id"]
            start_node_id = task["start_node"]
            end_node_id = task["end_node"]
            current_state = task.get("current_state", {})

            if start_node_id not in self.nodes or end_node_id not in self.nodes:
                logger.error(
                    f"Task {task_id}: Start node {start_node_id} or end node {end_node_id} not in graph."
                )
                results.append(
                    {
                        "task_id": task_id,
                        "path": [],
                        "cost": np.inf,
                        "status": "Invalid Nodes",
                    }
                )
                continue

            start_idx = self.node_map[start_node_id]
            end_idx = self.node_map[end_node_id]

            logger.info(
                f"Simulating quantum-inspired optimization for task {task_id} ({start_node_id} -> {end_node_id})..."
            )
            result = self._simulate_one_quantum_annealing_task(
                start_idx, end_idx, current_state
            )

            if result["cost"] == np.inf:
                results.append(
                    {
                        "task_id": task_id,
                        "path": [],
                        "cost": np.inf,
                        "status": "No Path Found",
                    }
                )
            else:
                results.append({**result, "task_id": task_id, "status": "Success"})

            self.simulated_paths.append(result)

        logger.info(f"Completed {len(results)} quantum-inspired optimization tasks.")
        return results


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    graph_file = "data_nexus/road_network_graph/q_sim_test_graph.gml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)

    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    quantum_simulator:
      enabled: true
      max_concurrent_tasks: 3
      num_qubits_per_task: 10
      annealing_iterations: 100
      initial_temperature: 100.0
      cooling_rate: 0.95
      history_size: 50
      cost_weights:
        travel_time: 1.0
        load_penalty: 0.2
        driver_fatigue_penalty: 0.5
        demand_incentive: -0.3
"""
            )

    # Create a test graph for demonstration
    G_test = nx.MultiDiGraph()
    G_test.add_edges_from(
        [
            (0, 1, {"travel_time": 10, "traffic_factor": 1.0}),
            (0, 2, {"travel_time": 50, "traffic_factor": 1.0}),
            (1, 3, {"travel_time": 20, "traffic_factor": 1.2}),
            (1, 4, {"travel_time": 70, "traffic_factor": 1.0}),
            (2, 4, {"travel_time": 15, "traffic_factor": 0.8}),
            (2, 5, {"travel_time": 80, "traffic_factor": 1.5}),
            (3, 6, {"travel_time": 30, "traffic_factor": 1.0}),
            (4, 6, {"travel_time": 10, "traffic_factor": 1.1}),
            (5, 6, {"travel_time": 25, "traffic_factor": 0.9}),
        ],
        key=0,
    )
    # Add some node attributes for demand density simulation
    G_test.nodes[2]["demand_density"] = 0.8
    G_test.nodes[4]["demand_density"] = 0.6
    nx.write_gml(G_test, graph_file)
    print("Test graph generated.")

    q_simulator = QuantumOptimizationSimulator(config_file, graph_path=graph_file)

    # Define some simulation tasks
    tasks_to_simulate = [
        {
            "task_id": "route_driver_A",
            "start_node": 0,
            "end_node": 6,
            "current_state": {
                "driver_current_load": 0.0,
                "driver_fatigue_score": 0.1,
                "node_demand_density": G_test.nodes[2].get(
                    "demand_density", 0.0
                ),  # Relevant for path choice
            },
        },
        {
            "task_id": "route_driver_B",
            "start_node": 0,
            "end_node": 6,
            "current_state": {
                "driver_current_load": 50.0,  # High load penalty
                "driver_fatigue_score": 0.7,  # High fatigue penalty
                "node_demand_density": G_test.nodes[4].get("demand_density", 0.0),
            },
        },
        {
            "task_id": "urgent_delivery_C",
            "start_node": 1,
            "end_node": 5,
            "current_state": {
                "driver_current_load": 10.0,
                "driver_fatigue_score": 0.3,
                "node_demand_density": 0.1,  # Low demand incentive
            },
        },
    ]

    results = q_simulator.simulate_optimization(tasks_to_simulate)

    print("\n--- Simulation Results ---")
    for res in results:
        print(
            f"Task {res['task_id']}: Status={res['status']}, Path={res['path']}, Cost={res['cost']:.2f}"
        )

    print("\nQuantum Optimization Simulator demo complete.")
