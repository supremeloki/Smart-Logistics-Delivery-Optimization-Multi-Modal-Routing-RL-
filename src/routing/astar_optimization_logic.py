import math
import pandas as pd
import yaml
import logging
import networkx as nx
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AStarRouting:
    def __init__(self, graph, config_path):
        self.graph = graph
        self.config = self._load_config(config_path)["astar_custom_params"]

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  # meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _heuristic(self, node, target_node):
        x1, y1 = self.graph.nodes[node]["x"], self.graph.nodes[node]["y"]
        x2, y2 = self.graph.nodes[target_node]["x"], self.graph.nodes[target_node]["y"]
        # Assuming projected coordinates, Euclidean distance is fine
        # If not, use Haversine based on 'y' (lat) and 'x' (lon)
        if (
            "y" in self.graph.nodes[node]
            and "x" in self.graph.nodes[node]
            and "y" in self.graph.nodes[target_node]
            and "x" in self.graph.nodes[target_node]
        ):
            return self._haversine_distance(y1, x1, y2, x2) / (
                self.config.get("avg_speed_kph", 40) / 3.6
            )  # Estimate time
        return 0  # Fallback

    def _cost(self, u, v, data, current_time=None, vehicle_properties=None):
        travel_time = data.get("travel_time", 60)  # Default to 1 minute
        traffic_factor = data.get("traffic_factor", 1.0)  # From real-time updates/GNN

        # Dynamic cost adjustments
        weighted_time_cost = (
            travel_time * traffic_factor * self.config.get("weight_factor_time", 1.2)
        )
        weighted_distance_cost = (
            data.get("length", 100)
            * self.config.get("weight_factor_distance", 1.0)
            / 1000
        )  # Convert to km cost

        total_cost = (
            weighted_time_cost + weighted_distance_cost
        )  # Example combined cost

        if (
            self.config.get("time_windows_enabled", False)
            and current_time is not None
            and vehicle_properties is not None
        ):
            # Placeholder for time window logic
            # If arriving at 'v' outside a critical time window for an associated order, add penalty
            pass

        return total_cost

    def find_path(
        self, origin_node, target_node, current_time=None, vehicle_properties=None
    ):
        try:
            path_cost, path = nx.astar_path(
                self.graph,
                source=origin_node,
                target=target_node,
                heuristic=lambda u, v: self._heuristic(u, v),
                weight=lambda u, v, data: self._cost(
                    u, v, data, current_time, vehicle_properties
                ),
            )
            return path, path_cost
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {origin_node} and {target_node}.")
            return None, float("inf")
        except Exception as e:
            logger.error(f"Error during A* pathfinding: {e}")
            return None, float("inf")


if __name__ == "__main__":
    # Example usage (requires a preprocessed graph)
    # Ensure data_nexus/road_network_graph/preprocessed_tehran_graph.gml exists
    try:
        graph = nx.read_gml(
            "data_nexus/road_network_graph/preprocessed_tehran_graph.gml"
        )
        # Ensure 'x', 'y', 'travel_time', 'length' are present
        if not graph.nodes or "x" not in next(iter(graph.nodes(data=True)))[1]:
            # Re-add dummy x, y if not available after loading from GML
            for i, node in enumerate(graph.nodes):
                graph.nodes[node]["x"] = random.uniform(51.2, 51.7)
                graph.nodes[node]["y"] = random.uniform(35.5, 35.9)
        for u, v, k, data in graph.edges(data=True, keys=True):
            if "travel_time" not in data:
                data["travel_time"] = random.uniform(30, 300)
            if "length" not in data:
                data["length"] = random.uniform(50, 1000)
            data["traffic_factor"] = random.uniform(0.8, 1.5)

        router = AStarRouting(graph, "conf/routing_engine_config.yaml")

        # Pick random origin and destination nodes
        nodes = list(graph.nodes())
        origin = random.choice(nodes)
        destination = random.choice(nodes)

        current_time = pd.Timestamp.now()
        vehicle_props = {"fuel_efficiency": 0.1}

        path, cost = router.find_path(origin, destination, current_time, vehicle_props)
        if path:
            logger.info(
                f"A* Path found from {origin} to {destination} with cost: {cost:.2f}"
            )
            logger.info(f"Path length: {len(path)} nodes")
        else:
            logger.info("A* Path not found.")
    except FileNotFoundError:
        logger.error("Graph file not found. Please run osmnx_processor.py first.")
    except Exception as e:
        logger.error(f"Error during A* example: {e}")
