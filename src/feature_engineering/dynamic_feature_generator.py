import datetime
import numpy as np
import redis
import json
import yaml
import os
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicFeatureGenerator:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.env_config = self.config["environments"][environment]
        self.redis_client = redis.StrictRedis(
            host=self.env_config["redis"]["host"],
            port=self.env_config["redis"]["port"],
            db=self.env_config["redis"]["db"],
            decode_responses=True,
        )
        self.telemetry_stream_key = self.env_config["kafka"].get(
            "topic_telemetry_data", "dev_telemetry_stream"
        )
        self.order_stream_key = self.env_config["kafka"].get(
            "topic_order_data", "dev_order_events"
        )

        self.driver_window_data = {}  # driver_id -> {metric_name: deque}
        self.order_window_data = {}  # node_id -> {metric_name: deque}
        self.window_size_minutes = 5

        self.consumer_group_telemetry = "feature_gen_telemetry_group"
        self.consumer_group_orders = "feature_gen_orders_group"
        self.consumer_name = f"feature_gen_instance_{os.getpid()}"

        self._initialize_stream_consumer_group(
            self.telemetry_stream_key, self.consumer_group_telemetry
        )
        self._initialize_stream_consumer_group(
            self.order_stream_key, self.consumer_group_orders
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
    kafka:
      bootstrap_servers: ['localhost:9092']
      topic_telemetry_data: dev_telemetry_stream
      topic_order_data: dev_order_events
    feature_store:
      prefix_driver_dynamic: driver_dynamic_features:
      prefix_node_demand: node_demand_features:
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _initialize_stream_consumer_group(self, stream_key, group_name):
        try:
            self.redis_client.xgroup_create(stream_key, group_name, "0", mkstream=True)
            logger.info(
                f"Redis Stream Consumer Group '{group_name}' created for '{stream_key}'."
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(
                    f"Redis Stream Consumer Group '{group_name}' already exists."
                )
            else:
                logger.error(f"Error creating Redis Stream Consumer Group: {e}")
                raise

    def _process_telemetry_message(self, msg_id, data):
        driver_id = data.get("driver_id")
        timestamp_str = data.get("timestamp")
        if not driver_id or not timestamp_str:
            logger.warning(f"Malformed telemetry message: {data}")
            return

        msg_timestamp = datetime.datetime.fromisoformat(
            timestamp_str.replace("Z", "+00:00")
        )
        current_time_seconds = msg_timestamp.timestamp()

        if driver_id not in self.driver_window_data:
            self.driver_window_data[driver_id] = {
                "speed_kph": deque(),
                "load_kg": deque(),
                "timestamp": deque(),
            }

        self.driver_window_data[driver_id]["speed_kph"].append(
            (current_time_seconds, float(data.get("speed_kph", 0)))
        )
        self.driver_window_data[driver_id]["load_kg"].append(
            (current_time_seconds, float(data.get("load_kg", 0)))
        )
        self.driver_window_data[driver_id]["timestamp"].append(current_time_seconds)

        # Trim old data
        window_start_time = current_time_seconds - (self.window_size_minutes * 60)
        for metric, deq in self.driver_window_data[driver_id].items():
            while deq and deq[0][0] < window_start_time:
                deq.popleft()

    def _generate_driver_features(self):
        prefix = self.config["environments"]["dev"]["feature_store"][
            "prefix_driver_dynamic"
        ]
        for driver_id, data in self.driver_window_data.items():
            speeds = [val for ts, val in data["speed_kph"]]
            loads = [val for ts, val in data["load_kg"]]
            list(data["timestamp"])

            if not speeds:
                continue

            current_time = datetime.datetime.utcnow().isoformat()

            features = {
                "timestamp_utc": current_time,
                "avg_speed_5min": round(np.mean(speeds), 2) if speeds else 0,
                "std_speed_5min": round(np.std(speeds), 2) if speeds else 0,
                "avg_load_5min": round(np.mean(loads), 2) if loads else 0,
                "last_speed": speeds[-1] if speeds else 0,
                "last_load": loads[-1] if loads else 0,
                "num_telemetry_5min": len(speeds),
            }
            self.redis_client.set(f"{prefix}{driver_id}", json.dumps(features))
            logger.debug(
                f"Generated dynamic features for driver {driver_id}: {features}"
            )

    def _process_order_message(self, msg_id, data):
        origin_node = data.get("origin_node")
        timestamp_str = data.get("pickup_time_start")
        if not origin_node or not timestamp_str:
            logger.warning(f"Malformed order message: {data}")
            return

        msg_timestamp = datetime.datetime.fromisoformat(
            timestamp_str.replace("Z", "+00:00")
        )
        current_time_seconds = msg_timestamp.timestamp()

        if origin_node not in self.order_window_data:
            self.order_window_data[origin_node] = {
                "order_count": deque(),
                "total_weight": deque(),
                "total_volume": deque(),
                "timestamp": deque(),
            }

        self.order_window_data[origin_node]["order_count"].append(
            (current_time_seconds, 1)
        )
        self.order_window_data[origin_node]["total_weight"].append(
            (current_time_seconds, float(data.get("weight", 0)))
        )
        self.order_window_data[origin_node]["total_volume"].append(
            (current_time_seconds, float(data.get("volume", 0)))
        )
        self.order_window_data[origin_node]["timestamp"].append(current_time_seconds)

        # Trim old data
        window_start_time = current_time_seconds - (self.window_size_minutes * 60)
        for metric, deq in self.order_window_data[origin_node].items():
            while deq and deq[0][0] < window_start_time:
                deq.popleft()

    def _generate_node_demand_features(self):
        prefix = self.config["environments"]["dev"]["feature_store"][
            "prefix_node_demand"
        ]
        for node_id, data in self.order_window_data.items():
            order_counts = [val for ts, val in data["order_count"]]
            total_weights = [val for ts, val in data["total_weight"]]
            total_volumes = [val for ts, val in data["total_volume"]]

            if not order_counts:
                continue

            current_time = datetime.datetime.utcnow().isoformat()

            features = {
                "timestamp_utc": current_time,
                "num_new_orders_5min": sum(order_counts),
                "total_weight_5min": round(sum(total_weights), 2),
                "total_volume_5min": round(sum(total_volumes), 2),
            }
            self.redis_client.set(f"{prefix}{node_id}", json.dumps(features))
            logger.debug(f"Generated dynamic features for node {node_id}: {features}")

    async def run_feature_generation_loop(
        self, poll_interval_ms=1000, max_messages_per_read=50
    ):
        logger.info("Starting dynamic feature generation loop...")
        last_id_telemetry = "0-0"
        last_id_orders = "0-0"

        while True:
            try:
                # Process telemetry stream
                response_telemetry = self.redis_client.xreadgroup(
                    self.consumer_group_telemetry,
                    self.consumer_name,
                    {self.telemetry_stream_key: last_id_telemetry},
                    count=max_messages_per_read,
                    block=poll_interval_ms // 2,
                )
                if response_telemetry:
                    for stream_name, messages in response_telemetry:
                        for msg_id, msg_data_dict in messages:
                            self._process_telemetry_message(msg_id, msg_data_dict)
                        self.redis_client.xack(
                            self.telemetry_stream_key,
                            self.consumer_group_telemetry,
                            *[m[0] for m in messages],
                        )
                        last_id_orders = messages[-1][0]

                # Process order stream
                response_orders = self.redis_client.xreadgroup(
                    self.consumer_group_orders,
                    self.consumer_name,
                    {self.order_stream_key: last_id_orders},
                    count=max_messages_per_read,
                    block=poll_interval_ms // 2,
                )
                if response_orders:
                    for stream_name, messages in response_orders:
                        for msg_id, msg_data_dict in messages:
                            self._process_order_message(msg_id, msg_data_dict)
                        self.redis_client.xack(
                            self.order_stream_key,
                            self.consumer_group_orders,
                            *[m[0] for m in messages],
                        )
                        last_id_orders = messages[-1][0]

                # Generate and store aggregated features
                self._generate_driver_features()
                self._generate_node_demand_features()

                await asyncio.sleep(poll_interval_ms / 1000.0)

            except Exception as e:
                logger.error(
                    f"Error in dynamic feature generation loop: {e}", exc_info=True
                )
                await asyncio.sleep(5)


if __name__ == "__main__":
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
    kafka:
      bootstrap_servers: ['localhost:9092']
      topic_telemetry_data: dev_telemetry_stream
      topic_order_data: dev_order_events
    feature_store:
      prefix_driver_dynamic: driver_dynamic_features:
      prefix_node_demand: node_demand_features:
"""
            )

    # This requires Redis to be running locally and KafkaProducerSimulator to push data
    # `docker-compose -f deployment_ops/docker-compose.yaml up -d redis kafka`
    # Then run src/stream_processing/kafka_producer_simulator.py in a separate terminal.

    async def main():
        feature_gen = DynamicFeatureGenerator(config_file)
        print("Starting DynamicFeatureGenerator for 60 seconds...")
        await asyncio.sleep(5)  # Give producer time to push some events
        try:
            await asyncio.wait_for(
                feature_gen.run_feature_generation_loop(), timeout=60
            )
        except asyncio.TimeoutError:
            print("DynamicFeatureGenerator timed out after 60 seconds.")
        except KeyboardInterrupt:
            print("DynamicFeatureGenerator stopped by user.")

        print("\nChecking generated features in Redis:")
        for key in feature_gen.redis_client.scan_iter("driver_dynamic_features:*"):
            print(f"{key}: {feature_gen.redis_client.get(key)}")
        for key in feature_gen.redis_client.scan_iter("node_demand_features:*"):
            print(f"{key}: {feature_gen.redis_client.get(key)}")

    import asyncio

    asyncio.run(main())
