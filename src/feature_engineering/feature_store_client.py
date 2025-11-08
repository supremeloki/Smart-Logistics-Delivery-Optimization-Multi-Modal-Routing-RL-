import redis
import json
import datetime
import yaml
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStoreClient:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.redis_config = self.config["environments"][environment]["redis"]

        self.client = redis.StrictRedis(
            host=self.redis_config["host"],
            port=self.redis_config["port"],
            db=self.redis_config["db"],
            decode_responses=True,
        )

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
    feature_store:
      prefix_traffic: traffic_edge:
      prefix_driver: driver_feature:
      prefix_order: order_feature:
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_key_prefix(self, feature_group):
        prefix_config = self.config["environments"]["dev"]["feature_store"]
        return prefix_config.get(f"prefix_{feature_group}", f"{feature_group}_feature:")

    def get_feature(self, feature_group: str, entity_id: str):
        key = f"{self._get_key_prefix(feature_group)}{entity_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def set_feature(
        self,
        feature_group: str,
        entity_id: str,
        features: dict,
        ttl_seconds: int = None,
    ):
        key = f"{self._get_key_prefix(feature_group)}{entity_id}"
        self.client.set(key, json.dumps(features))
        if ttl_seconds:
            self.client.expire(key, ttl_seconds)

    def get_multiple_features(self, feature_group: str, entity_ids: list):
        keys = [f"{self._get_key_prefix(feature_group)}{eid}" for eid in entity_ids]
        data_list = self.client.mget(keys)
        results = {}
        for i, entity_id in enumerate(entity_ids):
            if data_list[i]:
                results[entity_id] = json.loads(data_list[i])
            else:
                results[entity_id] = None
        return results

    def stream_features_to_df(
        self, feature_group: str, pattern: str = "*", limit: int = 100
    ) -> pd.DataFrame:
        keys = self.client.keys(f"{self._get_key_prefix(feature_group)}{pattern}")
        if not keys:
            return pd.DataFrame()

        features_data = []
        for key in keys[:limit]:
            entity_id = key.split(self._get_key_prefix(feature_group))[1]
            data = self.client.get(key)
            if data:
                feature_dict = json.loads(data)
                feature_dict["entity_id"] = entity_id
                features_data.append(feature_dict)
        return pd.DataFrame(features_data)


if __name__ == "__main__":
    # Ensure a Redis instance is running on localhost:6379
    # docker run --name some-redis -p 6379:6379 -d redis

    client = FeatureStoreClient()

    # Set some dummy features
    client.set_feature(
        "traffic",
        "123-456-0",
        {
            "current_travel_time": 120,
            "historic_avg": 90,
            "timestamp": str(datetime.datetime.utcnow()),
        },
        ttl_seconds=300,
    )
    client.set_feature(
        "traffic",
        "789-012-1",
        {
            "current_travel_time": 60,
            "historic_avg": 70,
            "timestamp": str(datetime.datetime.utcnow()),
        },
        ttl_seconds=300,
    )
    client.set_feature(
        "driver",
        "driver_A",
        {"speed": 15, "load": 5, "status": "active"},
        ttl_seconds=600,
    )
    client.set_feature(
        "driver",
        "driver_B",
        {"speed": 10, "load": 2, "status": "idle"},
        ttl_seconds=600,
    )

    print("--- Getting individual features ---")
    traffic_feature = client.get_feature("traffic", "123-456-0")
    print(f"Traffic feature for 123-456-0: {traffic_feature}")
    driver_feature = client.get_feature("driver", "driver_A")
    print(f"Driver feature for driver_A: {driver_feature}")

    print("\n--- Getting multiple features ---")
    multiple_traffic = client.get_multiple_features(
        "traffic", ["123-456-0", "non_existent_edge", "789-012-1"]
    )
    print(f"Multiple traffic features: {multiple_traffic}")
    multiple_drivers = client.get_multiple_features(
        "driver", ["driver_A", "driver_B", "driver_C"]
    )
    print(f"Multiple driver features: {multiple_drivers}")

    print("\n--- Streaming features to DataFrame ---")
    traffic_df = client.stream_features_to_df("traffic")
    print("Traffic DataFrame:")
    print(traffic_df)

    driver_df = client.stream_features_to_df("driver")
    print("\nDriver DataFrame:")
    print(driver_df)

    # Test TTL
    client.set_feature("test", "ephemeral", {"data": "will vanish"}, ttl_seconds=5)
    print("\nEphemeral feature set. Waiting 6 seconds...")
    import time

    time.sleep(6)
    ephemeral_feature = client.get_feature("test", "ephemeral")
    print(f"Ephemeral feature after TTL: {ephemeral_feature}")
