import json
import yaml
import os
import logging
import random
import asyncio
from typing import Dict, Any, List
import redis
import networkx as nx
import numpy as np


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "traffic_prediction:node_1:node_2:current",
            json.dumps({"travel_time_factor": 1.2, "congestion_level": "medium"}),
        )
        self.client.set(
            "weather_impact:node_2:node_3:current",
            json.dumps({"travel_time_factor": 1.1, "condition": "light_rain"}),
        )
        self.client.set(
            "driver_wellbeing:driver_A",
            json.dumps({"fatigue_score": 0.3, "stress_score": 0.4}),
        )
        self.client.set(
            "order:ORD_123",
            json.dumps(
                {
                    "origin_node": "node_1",
                    "destination_node": "node_4",
                    "priority": 5,
                    "payload_weight": 10,
                }
            ),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))


class MockDRLAgent:
    def __init__(
        self, model_path: str, observation_space_dim: int, action_space_size: int
    ):
        self.model_path = model_path
        self.observation_space_dim = observation_space_dim
        self.action_space_size = action_space_size

    def get_action_sequence(
        self, observation: np.ndarray, current_path_length: int
    ) -> List[int]:
        if random.random() < 0.2:  # Simulate early termination or complex path
            return [
                random.randint(0, self.action_space_size - 1)
                for _ in range(random.randint(3, 5))
            ]
        return [
            random.randint(0, self.action_space_size - 1)
            for _ in range(min(self.action_space_size - 1, current_path_length + 2))
        ]

    def load_model(self):
        pass


class MockOSMNxProcessor:
    def __init__(self, graph_path: str = None):
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(
            [
                ("node_1", {"lat": 35.7, "lon": 51.4}),
                ("node_2", {"lat": 35.71, "lon": 51.41}),
                ("node_3", {"lat": 35.72, "lon": 51.42}),
                ("node_4", {"lat": 35.73, "lon": 51.43}),
            ]
        )
        self.graph.add_edges_from(
            [
                ("node_1", "node_2", 0, {"travel_time_base": 60, "length_km": 1.0}),
                ("node_2", "node_3", 0, {"travel_time_base": 50, "length_km": 0.8}),
                ("node_3", "node_4", 0, {"travel_time_base": 70, "length_km": 1.2}),
                ("node_1", "node_3", 0, {"travel_time_base": 110, "length_km": 1.8}),
                ("node_2", "node_4", 0, {"travel_time_base": 100, "length_km": 1.5}),
            ]
        )
        self.node_to_idx = {node: i for i, node in enumerate(self.graph.nodes())}
        self.idx_to_node = {i: node for i, node in enumerate(self.graph.nodes())}

    def get_graph(self):
        return self.graph

    def get_node_id_from_coords(self, lat, lon):
        return f"node_{random.randint(1, 4)}"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DRLPredictiveRouter:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.router_config = self.config["environments"][environment][
            "drl_predictive_router"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.osm_processor = MockOSMNxProcessor(self.router_config["graph_path"])
        self.graph = self.osm_processor.get_graph()

        self.drl_agent = MockDRLAgent(
            self.router_config["model_path"],
            self.router_config["observation_space_dim"],
            len(self.graph.nodes()),  # Action space maps to next node
        )
        self.drl_agent.load_model()
        logger.info("DRLPredictiveRouter initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    drl_predictive_router:
      enabled: true
      routing_interval_seconds: 10
      model_path: ml_models/drl_router_model.pth
      graph_path: data_nexus/road_network_graph/preprocessed_tehran_graph.gml
      observation_space_dim: 256
      min_re_route_delta_percent: 0.1 # Min improvement needed to re-route
      cost_weights:
        travel_time: 1.0
        fuel_consumption: 0.5
        driver_fatigue: 0.2
        delivery_priority: -0.8 # Negative for incentive
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _generate_observation(
        self,
        current_node: str,
        destination_node: str,
        driver_id: str = None,
        order_data: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Constructs an observation vector for the DRL agent."""
        # This observation would include:
        # - Current node & destination node embeddings (if GNN-based graph)
        # - Real-time traffic data for surrounding edges
        # - Weather conditions
        # - Driver's current fatigue/stress levels
        # - Order priority/payload
        # - Time of day / historical patterns

        obs_vec = np.zeros(self.router_config["observation_space_dim"])

        # Mock node features (embedding)
        current_node_idx = self.osm_processor.node_to_idx.get(current_node, 0)
        dest_node_idx = self.osm_processor.node_to_idx.get(destination_node, 0)
        obs_vec[0] = current_node_idx / len(self.graph.nodes())
        obs_vec[1] = dest_node_idx / len(self.graph.nodes())

        # Mock traffic data for relevant edges
        # In a real scenario, this would aggregate data for multiple edges around current/dest
        traffic_data = self.feature_store.get_feature(
            "traffic_prediction", f"{current_node}:{destination_node}:current"
        )
        obs_vec[2] = traffic_data.get("travel_time_factor", 1.0)

        # Mock driver wellbeing
        if driver_id:
            driver_wellbeing = self.feature_store.get_feature(
                "driver_wellbeing", driver_id
            )
            obs_vec[3] = driver_wellbeing.get("fatigue_score", 0.0)
            obs_vec[4] = driver_wellbeing.get("stress_score", 0.0)

        # Mock order data
        if order_data:
            obs_vec[5] = order_data.get("priority", 0) / 10.0  # Normalize priority
            obs_vec[6] = order_data.get("payload_weight", 0) / 100.0  # Normalize weight

        return obs_vec  # Return the mock observation vector

    def _calculate_path_cost(
        self, path: List[str], driver_id: str = None, order_data: Dict[str, Any] = None
    ) -> float:
        """Calculates the total cost of a given path considering dynamic factors."""
        total_cost = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.graph.get_edge_data(
                u, v, key=0
            )  # Assuming key=0 for primary edge
            if not edge_data:
                return np.inf  # Invalid path segment

            travel_time_base = edge_data.get("travel_time_base", 60)  # seconds
            length_km = edge_data.get("length_km", 1.0)

            # Dynamic factors
            traffic_data = self.feature_store.get_feature(
                "traffic_prediction", f"{u}:{v}:current"
            )
            weather_data = self.feature_store.get_feature(
                "weather_impact", f"{u}:{v}:current"
            )

            current_travel_time = (
                travel_time_base
                * traffic_data.get("travel_time_factor", 1.0)
                * weather_data.get("travel_time_factor", 1.0)
            )
            total_cost += (
                current_travel_time * self.router_config["cost_weights"]["travel_time"]
            )

            # Mock fuel consumption cost (e.g., higher for longer routes)
            total_cost += (
                length_km * self.router_config["cost_weights"]["fuel_consumption"]
            )

        # Driver-specific costs
        if driver_id:
            wellbeing_data = self.feature_store.get_feature(
                "driver_wellbeing", driver_id
            )
            fatigue_impact = (
                wellbeing_data.get("fatigue_score", 0.0)
                * self.router_config["cost_weights"]["driver_fatigue"]
            )
            stress_impact = (
                wellbeing_data.get("stress_score", 0.0)
                * self.router_config["cost_weights"]["driver_fatigue"]
            )  # Using same weight for demo
            total_cost += (fatigue_impact + stress_impact) * (
                total_cost / travel_time_base if travel_time_base > 0 else 1.0
            )  # Scale by route length

        # Order priority as negative cost (incentive)
        if order_data:
            total_cost -= (
                order_data.get("priority", 0)
                * self.router_config["cost_weights"]["delivery_priority"]
            )

        return total_cost

    async def generate_optimized_route(
        self, driver_id: str, order_id: str
    ) -> Dict[str, Any]:
        """
        Generates an optimized route for a specific driver and order using DRL.
        """
        if not self.router_config["enabled"]:
            logger.info("DRL Predictive Router is disabled.")
            return {"status": "DISABLED", "path": [], "cost": np.inf}

        order_data = self.feature_store.get_feature("order", order_id)
        if not order_data:
            logger.error(f"Order {order_id} not found for routing.")
            return {
                "status": "ERROR",
                "message": "Order not found",
                "path": [],
                "cost": np.inf,
            }

        start_node = order_data["origin_node"]
        end_node = order_data["destination_node"]

        current_path_nodes = [start_node]
        current_node = start_node
        path_cost = 0.0

        # Build path step by step using DRL
        for _ in range(
            self.router_config["observation_space_dim"]
        ):  # Max path length as a heuristic limit
            if current_node == end_node:
                break

            observation = self._generate_observation(
                current_node, end_node, driver_id, order_data
            )
            action_indices = self.drl_agent.get_action_sequence(
                observation, len(current_path_nodes)
            )  # Get sequence for potential path

            next_node = None
            for action_idx in action_indices:
                possible_next_node_id = self.osm_processor.idx_to_node.get(
                    action_idx % len(self.graph.nodes())
                )  # Map action to a node ID
                if possible_next_node_id and self.graph.has_edge(
                    current_node, possible_next_node_id
                ):
                    next_node = possible_next_node_id
                    break

            if not next_node:
                logger.warning(
                    f"DRL agent could not find a valid next step from {current_node}. Falling back to shortest path if possible."
                )
                try:  # Fallback to Dijkstra for robustness
                    remaining_path = nx.shortest_path(
                        self.graph,
                        source=current_node,
                        target=end_node,
                        weight="travel_time_base",
                    )
                    current_path_nodes.extend(remaining_path[1:])
                    break
                except nx.NetworkXNoPath:
                    logger.error(f"No path found from {current_node} to {end_node}.")
                    return {
                        "status": "NO_PATH",
                        "path": current_path_nodes,
                        "cost": np.inf,
                    }

            path_cost += self._calculate_path_cost(
                [current_node, next_node], driver_id, order_data
            )
            current_path_nodes.append(next_node)
            current_node = next_node

        if current_path_nodes[-1] != end_node:
            logger.warning(
                f"DRL agent did not reach destination {end_node}. Final path: {current_path_nodes}"
            )
            return {
                "status": "INCOMPLETE_PATH",
                "path": current_path_nodes,
                "cost": np.inf,
            }

        logger.info(
            f"Generated DRL-optimized path for Order {order_id} (Driver {driver_id}): {current_path_nodes}, Cost: {path_cost:.2f}"
        )
        return {"status": "SUCCESS", "path": current_path_nodes, "cost": path_cost}

    async def run_routing_loop(self):
        """Monitors for pending orders and generates/updates routes."""
        logger.info("Starting DRLPredictiveRouter loop...")
        while True:
            try:
                # In a real system, this would trigger when an order is assigned or real-time conditions change
                # For demo, let's process a mock pending order
                pending_orders = self.feature_store.get_all_features_by_group(
                    "order", pattern="NEW_ORDER_*"
                )
                for order_id, order_data in pending_orders.items():
                    if (
                        order_data.get("status") == "pending_routing"
                    ):  # A specific status for routing
                        # Assign a dummy driver for now
                        driver_id = "driver_A"
                        routing_result = await self.generate_optimized_route(
                            driver_id, order_id
                        )
                        if routing_result["status"] == "SUCCESS":
                            order_data["current_route"] = routing_result["path"]
                            order_data["estimated_cost"] = routing_result["cost"]
                            order_data["status"] = "route_assigned"
                            self.feature_store.set_feature(
                                "order", order_id, order_data
                            )
                            logger.info(
                                f"New route for Order {order_id} assigned: {routing_result['path']}"
                            )
                        else:
                            logger.error(
                                f"Failed to route Order {order_id}: {routing_result['message']}"
                            )
                    elif order_data.get("status") == "on_route":
                        # Periodically re-evaluate route due to dynamic changes (traffic, weather)
                        # For demo, just log, actual re-routing logic would be here
                        logger.debug(
                            f"Re-evaluating route for Order {order_id} (on route)."
                        )
                        # You would re-run generate_optimized_route and compare cost/ETA.
                        # If significant change, update the driver/AV.

            except Exception as e:
                logger.error(f"Error in DRL Predictive Router loop: {e}", exc_info=True)

            await asyncio.sleep(self.router_config["routing_interval_seconds"])


if __name__ == "__main__":
    import redis

    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    drl_predictive_router:
      enabled: true
      routing_interval_seconds: 5
      model_path: ml_models/drl_router_model.pth
      graph_path: data_nexus/road_network_graph/preprocessed_tehran_graph.gml
      observation_space_dim: 256
      min_re_route_delta_percent: 0.1
      cost_weights:
        travel_time: 1.0
        fuel_consumption: 0.5
        driver_fatigue: 0.2
        delivery_priority: -0.8
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis. Populating dummy data for DRLPredictiveRouter.")
        r.set(
            "order:NEW_ORDER_001",
            json.dumps(
                {
                    "origin_node": "node_1",
                    "destination_node": "node_4",
                    "priority": 5,
                    "payload_weight": 10,
                    "status": "pending_routing",
                }
            ),
        )
        r.set(
            "driver_wellbeing:driver_A",
            json.dumps({"fatigue_score": 0.3, "stress_score": 0.4}),
        )
        r.set(
            "traffic_prediction:node_1:node_2:current",
            json.dumps({"travel_time_factor": 1.2, "congestion_level": "medium"}),
        )
        r.set(
            "traffic_prediction:node_2:node_3:current",
            json.dumps({"travel_time_factor": 1.0, "congestion_level": "low"}),
        )
        r.set(
            "traffic_prediction:node_3:node_4:current",
            json.dumps({"travel_time_factor": 1.1, "congestion_level": "light"}),
        )
        r.set(
            "traffic_prediction:node_1:node_3:current",
            json.dumps({"travel_time_factor": 1.5, "congestion_level": "high"}),
        )  # Make direct route more costly
    except redis.exceptions.ConnectionError:
        print("Redis not running. DRLPredictiveRouter will start with empty data.")

    async def main_drl_router():
        router = DRLPredictiveRouter(config_file)
        print("Starting DRLPredictiveRouter for 20 seconds...")
        try:
            await asyncio.wait_for(router.run_routing_loop(), timeout=20)
        except asyncio.TimeoutError:
            print("\nDRLPredictiveRouter demo timed out after 20 seconds.")
        except KeyboardInterrupt:
            print("\nDRLPredictiveRouter demo stopped by user.")

    asyncio.run(main_drl_router())
