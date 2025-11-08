import redis
import json
import logging
import yaml
import numpy as np
import math
import osmnx as ox
import random
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeTrafficFeatures:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.redis_client = redis.StrictRedis(
            host=self.config["redis"]["host"],
            port=self.config["redis"]["port"],
            db=self.config["redis"]["db"],
            decode_responses=True,
        )
        self.traffic_key_prefix = "traffic_edge:"

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def update_traffic_data(self, u, v, key, current_travel_time, timestamp):
        edge_id = f"{u}-{v}-{key}"
        data = {
            "current_travel_time": current_travel_time,
            "timestamp": timestamp.isoformat(),
        }
        self.redis_client.set(f"{self.traffic_key_prefix}{edge_id}", json.dumps(data))
        logger.debug(f"Updated traffic for edge {edge_id}")

    def get_traffic_factor(self, u, v, key, default_travel_time=None):
        edge_id = f"{u}-{v}-{key}"
        cached_data = self.redis_client.get(f"{self.traffic_key_prefix}{edge_id}")
        if cached_data:
            data = json.loads(cached_data)
            current_travel_time = data["current_travel_time"]
            if default_travel_time:
                return current_travel_time / default_travel_time
            return current_travel_time
        return 1.0  # Default to no traffic impact

    def generate_traffic_grid_features(
        self, graph, current_timestamp, resolution_km=1.0
    ):
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

        traffic_grid = np.zeros((num_lat_bins, num_lon_bins))
        traffic_counts = np.zeros((num_lat_bins, num_lon_bins))

        for u, v, k, data in graph.edges(keys=True, data=True):
            if "geometry" in data:
                coords = list(data["geometry"].coords)
            else:
                coords = [
                    (graph.nodes[u]["y"], graph.nodes[u]["x"]),
                    (graph.nodes[v]["y"], graph.nodes[v]["x"]),
                ]

            traffic_factor = self.get_traffic_factor(
                u, v, k, data.get("travel_time", 1)
            )

            for lat, lon in coords:
                lat_idx = min(
                    int((lat - lat_min) / (lat_max - lat_min) * num_lat_bins),
                    num_lat_bins - 1,
                )
                lon_idx = min(
                    int((lon - lon_min) / (lon_max - lon_min) * num_lon_bins),
                    num_lon_bins - 1,
                )

                traffic_grid[lat_idx, lon_idx] += traffic_factor
                traffic_counts[lat_idx, lon_idx] += 1

        traffic_grid_avg = np.divide(
            traffic_grid,
            traffic_counts,
            out=np.ones_like(traffic_grid),
            where=traffic_counts != 0,
        )
        return traffic_grid_avg

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
    import datetime

    # Requires a running Redis instance at localhost:6379
    # Requires conf/environments/dev.yaml
    processor = RealTimeTrafficFeatures("conf/environments/dev.yaml")

    # Create a dummy graph for feature generation
    place_name = "Tehran, Iran"
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Add bbox to graph if not present for feature generation
    if "bbox" not in G.graph:
        north, south, east, west = ox.utils_geo.get_bbox(G)
        G.graph["bbox"] = (south, west, north, east)

    # Simulate some traffic updates
    edges = list(G.edges(keys=True))
    for _ in range(50):
        u, v, k = random.choice(edges)
        travel_time_base = G[u][v][k].get("travel_time", 60)
        current_travel_time = travel_time_base * random.uniform(0.7, 2.5)
        timestamp = datetime.datetime.now() - datetime.timedelta(
            minutes=random.randint(0, 10)
        )
        processor.update_traffic_data(u, v, k, current_travel_time, timestamp)

    # Generate grid features
    traffic_grid = processor.generate_traffic_grid_features(G, datetime.datetime.now())
    logger.info(f"Generated traffic grid features with shape: {traffic_grid.shape}")
    logger.info(f"Sample grid values: {traffic_grid.flatten()[:10]}")
