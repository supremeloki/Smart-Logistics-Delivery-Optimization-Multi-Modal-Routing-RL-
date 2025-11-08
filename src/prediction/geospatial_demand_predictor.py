import datetime
import json
import yaml
import os
import logging
import random
import asyncio
from typing import Dict, Any
import redis
import networkx as nx
import pandas as pd
import numpy as np


# Mock components for a light sandbox environment
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        # Mock historical demand data
        self.client.set(
            "historical_demand:node_A:2025-10-20-10",
            json.dumps({"demand_count": 10, "peak_demand": 15}),
        )
        self.client.set(
            "historical_demand:node_A:2025-10-20-11",
            json.dumps({"demand_count": 12, "peak_demand": 18}),
        )
        self.client.set(
            "historical_demand:node_B:2025-10-20-10",
            json.dumps({"demand_count": 5, "peak_demand": 8}),
        )
        # Mock point-of-interest data
        self.client.set(
            "poi:node_A", json.dumps({"type": "commercial", "density": 0.8})
        )
        self.client.set(
            "poi:node_B", json.dumps({"type": "residential", "density": 0.5})
        )
        # Mock event data
        self.client.set(
            "event:concert_venue_node_C:2025-10-21-19",
            json.dumps({"type": "concert", "expected_attendees": 5000}),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def get_all_features_by_group(self, feature_group: str, pattern: str = "*"):
        keys = self.client.keys(f"{feature_group}:{pattern}")
        return (
            {k.split(":")[1]: json.loads(self.client.get(k)) for k in keys}
            if keys
            else {}
        )

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))


# Mock Graph Neural Network model (simplified)
class MockGNNPredictor:
    def __init__(self, graph: nx.Graph, node_features: pd.DataFrame):
        self.graph = graph
        self.node_features = node_features
        # Simulate training a simple model
        self.weights_time = random.uniform(0.1, 0.3)
        self.weights_spatial = random.uniform(0.2, 0.4)
        self.weights_poi = random.uniform(0.05, 0.15)
        logging.info("MockGNNPredictor initialized.")

    def predict_demand(
        self,
        target_node_id: str,
        current_time: datetime.datetime,
        lookahead_hours: int = 1,
    ) -> float:
        """Simulates a GNN-based demand prediction."""

        # 1. Temporal feature (e.g., hour of day)
        hour_of_day = current_time.hour
        time_feature = np.sin(2 * np.pi * hour_of_day / 24) * self.weights_time

        # 2. Local POI/density feature
        poi_data = (
            self.node_features.loc[target_node_id].get("poi_density", 0.5)
            if target_node_id in self.node_features.index
            else 0.5
        )
        local_poi_feature = poi_data * self.weights_poi

        # 3. Spatial aggregation (neighbors' historical demand)
        spatial_feature = 0
        if target_node_id in self.graph:
            neighbors = list(self.graph.neighbors(target_node_id))
            if neighbors:
                # Mock historical demand for neighbors
                neighbor_demands = [
                    random.uniform(0.1, 1.0) for _ in neighbors
                ]  # Simplified: random for sandbox
                spatial_feature = np.mean(neighbor_demands) * self.weights_spatial

        # 4. Event impact (simplified)
        event_impact = 0.0
        # For demo, if node C is a venue and there's a concert
        if (
            target_node_id == "node_C"
            and current_time.hour >= 18
            and current_time.hour <= 22
        ):
            event_impact = random.uniform(0.5, 1.0)  # Boost demand significantly

        # Combine features heuristically
        predicted_demand_score = np.clip(
            0.5 + time_feature + local_poi_feature + spatial_feature + event_impact,
            0.0,
            1.0,
        )
        return predicted_demand_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeospatialDemandPredictor:
    def __init__(
        self,
        config_path="conf/environments/prod.yaml",
        environment="dev",
        graph_path="data_nexus/city_graph.gml",
    ):
        self.config = self._load_config(config_path)
        self.demand_config = self.config["environments"][environment][
            "geospatial_demand_predictor"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.city_graph = self._load_or_generate_city_graph(graph_path)
        self.node_features = self._prepare_node_features()
        self.gnn_predictor = MockGNNPredictor(self.city_graph, self.node_features)

        logger.info("GeospatialDemandPredictor initialized.")

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
    geospatial_demand_predictor:
      enabled: true
      prediction_interval_seconds: 15
      prediction_lookahead_hours: 6
      demand_peak_threshold: 0.7
      city_graph_path: data_nexus/city_graph.gml
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_or_generate_city_graph(self, graph_path):
        if os.path.exists(graph_path):
            return nx.read_gml(graph_path)
        else:
            logger.warning(
                f"City graph not found at {graph_path}. Generating a small dummy graph."
            )
            G = nx.Graph()
            G.add_nodes_from(
                [
                    ("node_A", {"lat": 35.70, "lon": 51.40, "poi_density": 0.8}),
                    ("node_B", {"lat": 35.71, "lon": 51.41, "poi_density": 0.3}),
                    (
                        "node_C",
                        {
                            "lat": 35.72,
                            "lon": 51.42,
                            "poi_density": 0.9,
                            "is_event_venue": True,
                        },
                    ),
                    ("node_D", {"lat": 35.73, "lon": 51.43, "poi_density": 0.6}),
                ]
            )
            G.add_edges_from(
                [
                    ("node_A", "node_B", {"distance": 1.5}),
                    ("node_B", "node_C", {"distance": 1.0}),
                    ("node_C", "node_D", {"distance": 1.2}),
                    ("node_A", "node_D", {"distance": 2.0}),
                ]
            )
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            nx.write_gml(G, graph_path)
            return G

    def _prepare_node_features(self) -> pd.DataFrame:
        """Prepares node features (e.g., POI density, historical average demand) for the GNN."""
        features_list = []
        for node_id, data in self.city_graph.nodes(data=True):
            features = {
                "node_id": node_id,
                "lat": data.get("lat"),
                "lon": data.get("lon"),
            }

            # Add POI features directly from node data
            features["poi_density"] = data.get("poi_density", 0.0)
            features["is_event_venue"] = data.get("is_event_venue", False)

            # Mock historical average demand (in a real system, this would be aggregated)
            features["hist_avg_demand_daily"] = random.uniform(5, 20)

            features_list.append(features)

        return pd.DataFrame(features_list).set_index("node_id")

    async def predict_demand_for_all_nodes(self) -> Dict[str, Any]:
        """
        Predicts future demand for all relevant nodes in the city graph.
        """
        if not self.demand_config["enabled"]:
            logger.info("Geospatial demand predictor is disabled.")
            return {}

        predictions = {}
        current_time = datetime.datetime.utcnow()

        for node_id in self.city_graph.nodes():
            predicted_demand_score = self.gnn_predictor.predict_demand(
                node_id, current_time, self.demand_config["prediction_lookahead_hours"]
            )

            # Store in predictions
            predictions[node_id] = {
                "timestamp": current_time.isoformat() + "Z",
                "predicted_demand_score": round(predicted_demand_score, 3),
                "is_hotspot": predicted_demand_score
                > self.demand_config["demand_peak_threshold"],
            }
            self.feature_store.set_feature(
                "predicted_demand", node_id, predictions[node_id]
            )
            logger.debug(
                f"Node {node_id}: Predicted Demand = {predicted_demand_score:.3f}, Hotspot: {predictions[node_id]['is_hotspot']}"
            )

        logger.info(f"Generated demand predictions for {len(predictions)} nodes.")
        return predictions

    async def run_prediction_loop(self):
        """Main loop for periodically updating demand predictions."""
        if not self.demand_config["enabled"]:
            logger.info("Geospatial demand predictor is disabled.")
            return

        while True:
            try:
                await self.predict_demand_for_all_nodes()
            except Exception as e:
                logger.error(
                    f"Error in geospatial demand prediction loop: {e}", exc_info=True
                )

            await asyncio.sleep(self.demand_config["prediction_interval_seconds"])


if __name__ == "__main__":
    import redis

    config_file = "conf/environments/dev.yaml"
    graph_file = "data_nexus/city_graph.gml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
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
    geospatial_demand_predictor:
      enabled: true
      prediction_interval_seconds: 5 # Faster for demo
      prediction_lookahead_hours: 6
      demand_peak_threshold: 0.6 # Lower for easier trigger
      city_graph_path: data_nexus/city_graph.gml
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print(
            "Connected to Redis. Populating dummy data for GeospatialDemandPredictor."
        )
        # Ensure dummy historical data is present
        for node in ["node_A", "node_B", "node_C", "node_D"]:
            for i in range(24):  # Last 24 hours
                ts = (
                    datetime.datetime.utcnow() - datetime.timedelta(hours=i)
                ).strftime("%Y-%m-%d-%H")
                r.set(
                    f"historical_demand:{node}:{ts}",
                    json.dumps(
                        {
                            "demand_count": random.randint(0, 20),
                            "peak_demand": random.randint(0, 30),
                        }
                    ),
                )
        r.set(
            "poi:node_C",
            json.dumps({"type": "commercial", "density": 0.9, "is_event_venue": True}),
        )  # Node C as event venue
    except redis.exceptions.ConnectionError:
        print("Redis not running. Demand predictor will start with empty data.")

    async def main_demand_predictor():
        predictor = GeospatialDemandPredictor(config_file)
        print("Starting GeospatialDemandPredictor for 20 seconds...")
        try:
            await asyncio.wait_for(predictor.run_prediction_loop(), timeout=20)
        except asyncio.TimeoutError:
            print("\nGeospatialDemandPredictor demo timed out after 20 seconds.")
        except KeyboardInterrupt:
            print("\nGeospatialDemandPredictor demo stopped by user.")

    asyncio.run(main_demand_predictor())
