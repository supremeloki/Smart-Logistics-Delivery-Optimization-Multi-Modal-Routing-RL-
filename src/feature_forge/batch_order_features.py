import redis
import json
import logging
import yaml
import pandas as pd
import numpy as np
import math
import osmnx as ox
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchOrderFeatures:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.redis_client = redis.StrictRedis(
            host=self.config["redis"]["host"],
            port=self.config["redis"]["port"],
            db=self.config["redis"]["db"],
            decode_responses=True,
        )
        self.order_key_prefix = "order:"

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_active_orders(self):
        active_order_keys = self.redis_client.keys(f"{self.order_key_prefix}*")
        orders = []
        for key in active_order_keys:
            order_data = self.redis_client.get(key)
            if order_data:
                orders.append(json.loads(order_data))
        return pd.DataFrame(orders)

    def generate_order_density_features(self, orders_df, graph, resolution_km=1.0):
        if orders_df.empty:
            return np.zeros((10, 10))  # Default empty grid

        # Assuming graph has 'bbox' in its graph metadata
        if "bbox" not in graph.graph:
            logger.warning(
                "Graph does not have 'bbox'. Cannot generate density features."
            )
            return np.zeros((10, 10))

        lat_min, lon_min, lat_max, lon_max = graph.graph["bbox"]

        num_lat_bins = (
            int(
                self._haversine_distance(lat_min, lon_min, lat_max, lon_min)
                / 1000
                / resolution_km
            )
            + 1
        )
        num_lon_bins = (
            int(
                self._haversine_distance(lat_min, lon_min, lat_min, lon_max)
                / 1000
                / resolution_km
            )
            + 1
        )

        order_density_grid = np.zeros((num_lat_bins, num_lon_bins))

        gdf_nodes, _ = ox.graph_to_gdfs(graph)

        for _, order in orders_df.iterrows():
            pickup_node = order["origin_node"]
            delivery_node = order["destination_node"]

            if pickup_node in gdf_nodes.index:
                pickup_lat, pickup_lon = (
                    gdf_nodes.loc[pickup_node]["y"],
                    gdf_nodes.loc[pickup_node]["x"],
                )
                lat_idx = min(
                    int((pickup_lat - lat_min) / (lat_max - lat_min) * num_lat_bins),
                    num_lat_bins - 1,
                )
                lon_idx = min(
                    int((pickup_lon - lon_min) / (lon_max - lon_min) * num_lon_bins),
                    num_lon_bins - 1,
                )
                order_density_grid[lat_idx, lon_idx] += 1

            if delivery_node in gdf_nodes.index:
                delivery_lat, delivery_lon = (
                    gdf_nodes.loc[delivery_node]["y"],
                    gdf_nodes.loc[delivery_node]["x"],
                )
                lat_idx = min(
                    int((delivery_lat - lat_min) / (lat_max - lat_min) * num_lat_bins),
                    num_lat_bins - 1,
                )
                lon_idx = min(
                    int((delivery_lon - lon_min) / (lon_max - lon_min) * num_lon_bins),
                    num_lon_bins - 1,
                )
                order_density_grid[lat_idx, lon_idx] += 1

        return order_density_grid

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


if __name__ == "__main__":
    import random

    # Requires a running Redis instance at localhost:6379
    # Requires conf/environments/dev.yaml
    processor = BatchOrderFeatures("conf/environments/dev.yaml")

    # Create a dummy graph for feature generation
    place_name = "Tehran, Iran"
    G = ox.graph_from_place(place_name, network_type="drive")

    # Add bbox to graph if not present for feature generation
    if "bbox" not in G.graph:
        north, south, east, west = ox.utils_geo.get_bbox(G)
        G.graph["bbox"] = (south, west, north, east)

    nodes = list(G.nodes)

    # Simulate some orders in Redis
    for i in range(100):
        origin_node = random.choice(nodes)
        destination_node = random.choice(nodes)
        order_data = {
            "order_id": f"order_{i}",
            "origin_node": origin_node,
            "destination_node": destination_node,
            "status": "pending",
            "weight": random.uniform(0.5, 5.0),
        }
        processor.redis_client.set(f"order:order_{i}", json.dumps(order_data))

    # Load active orders and generate density features
    active_orders_df = processor.load_active_orders()
    order_density = processor.generate_order_density_features(active_orders_df, G)

    logger.info(f"Generated order density grid with shape: {order_density.shape}")
    logger.info(f"Sample density values: {order_density.flatten()[:10]}")
