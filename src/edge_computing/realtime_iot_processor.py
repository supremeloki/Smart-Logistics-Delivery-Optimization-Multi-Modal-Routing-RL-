import datetime
import json
import numpy as np
import yaml
import os
import logging
import random
import asyncio
from collections import deque
from typing import Any


# Mock connection to a local feature store/cache (e.g., SQLite, local Redis instance, or in-memory)
class MockLocalFeatureStore:
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.data = {}  # In-memory store
        logger.debug(f"MockLocalFeatureStore for {vehicle_id} initialized.")

    def set(self, key: str, value: Any, ttl_seconds: int = None):
        self.data[key] = {
            "value": value,
            "timestamp": datetime.datetime.utcnow(),
            "ttl": ttl_seconds,
        }
        # In a real local Redis, TTL would be managed by Redis.
        logger.debug(f"LocalFeatureStore set {key} for {self.vehicle_id}.")

    def get(self, key: str) -> Any:
        entry = self.data.get(key)
        if entry:
            if (
                entry["ttl"]
                and (datetime.datetime.utcnow() - entry["timestamp"]).total_seconds()
                > entry["ttl"]
            ):
                del self.data[key]
                return None
            return entry["value"]
        return None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeIoTProcessor:
    def __init__(
        self,
        vehicle_id: str,
        config_path="conf/environments/prod.yaml",
        environment="dev",
    ):
        self.vehicle_id = vehicle_id
        self.config = self._load_config(config_path)
        self.edge_config = self.config["environments"][environment]["edge_computing"]

        self.local_feature_store = MockLocalFeatureStore(vehicle_id)

        # Buffers for sensor data aggregation
        self.speed_buffer = deque(
            maxlen=self.edge_config["buffer_size_seconds"] * 2
        )  # Store for N seconds, 2 samples/sec
        self.engine_temp_buffer = deque(
            maxlen=self.edge_config["buffer_size_seconds"] * 2
        )
        self.gps_buffer = deque(maxlen=self.edge_config["buffer_size_seconds"] * 2)

        # Thresholds for local alerts/triggers
        self.max_speed_threshold = self.edge_config["max_speed_threshold_kph"]
        self.max_engine_temp_threshold = self.edge_config[
            "max_engine_temp_threshold_celsius"
        ]

        logger.info(f"RealtimeIoTProcessor for Vehicle {self.vehicle_id} initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    edge_computing:
      enabled: true
      buffer_size_seconds: 60 # Data window for local aggregation
      sampling_interval_ms: 500
      max_speed_threshold_kph: 80
      max_engine_temp_threshold_celsius: 105
      local_alert_debounce_seconds: 30
      cloud_sync_interval_seconds: 10
      telemetry_topic_cloud: dev_telemetry_stream # Kafka topic for cloud
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    async def _simulate_sensor_data(self):
        # Simulate continuous sensor data generation
        current_speed = random.uniform(0, 100)  # kph
        current_engine_temp = random.uniform(80, 110)  # Celsius
        current_lat, current_lon = random.uniform(35.5, 35.9), random.uniform(
            51.2, 51.7
        )

        while True:
            # Simulate dynamic changes
            current_speed += random.uniform(-5, 5)
            current_speed = np.clip(current_speed, 0, 120)

            current_engine_temp += random.uniform(-1, 2)
            current_engine_temp = np.clip(current_engine_temp, 70, 120)

            current_lat += random.uniform(-0.001, 0.001)
            current_lon += random.uniform(-0.001, 0.001)

            timestamp = datetime.datetime.utcnow()

            # Add to buffers
            self.speed_buffer.append((timestamp, current_speed))
            self.engine_temp_buffer.append((timestamp, current_engine_temp))
            self.gps_buffer.append((timestamp, (current_lat, current_lon)))

            # Trim old data from buffers
            window_start_time = timestamp - datetime.timedelta(
                seconds=self.edge_config["buffer_size_seconds"]
            )
            for buffer in [self.speed_buffer, self.engine_temp_buffer, self.gps_buffer]:
                while buffer and buffer[0][0] < window_start_time:
                    buffer.popleft()

            # Perform real-time processing and local alerts
            self._process_local_data()

            await asyncio.sleep(self.edge_config["sampling_interval_ms"] / 1000.0)

    def _process_local_data(self):
        current_time = datetime.datetime.utcnow()

        # Aggregated features for the current window
        current_speeds = [s for _, s in self.speed_buffer]
        current_temps = [t for _, t in self.engine_temp_buffer]

        if not current_speeds or not current_temps:
            return

        avg_speed = np.mean(current_speeds)
        max_engine_temp = np.max(current_temps)
        current_gps = self.gps_buffer[-1][1] if self.gps_buffer else (0, 0)

        # Store in local feature store for low-latency retrieval by local apps
        self.local_feature_store.set("avg_speed_kph", avg_speed)
        self.local_feature_store.set("max_engine_temp_celsius", max_engine_temp)
        self.local_feature_store.set("current_gps", current_gps)

        # Local decision-making / alerts
        if avg_speed > self.max_speed_threshold:
            last_alert = self.local_feature_store.get("last_speeding_alert_time")
            if (
                not last_alert
                or (current_time - last_alert).total_seconds()
                > self.edge_config["local_alert_debounce_seconds"]
            ):
                logger.warning(
                    f"[{self.vehicle_id} LOCAL ALERT] Excessive speed detected: {avg_speed:.1f} kph!"
                )
                self.local_feature_store.set("last_speeding_alert_time", current_time)

        if max_engine_temp > self.max_engine_temp_threshold:
            last_alert = self.local_feature_store.get("last_engine_temp_alert_time")
            if (
                not last_alert
                or (current_time - last_alert).total_seconds()
                > self.edge_config["local_alert_debounce_seconds"]
            ):
                logger.critical(
                    f"[{self.vehicle_id} LOCAL ALERT] Critical engine temperature: {max_engine_temp:.1f}C! Shutting down or warning driver."
                )
                self.local_feature_store.set(
                    "last_engine_temp_alert_time", current_time
                )
                # In a real system, this could trigger vehicle safety protocols.

    async def _sync_to_cloud_loop(self):
        # This function would send aggregated/processed data to a Kafka topic for cloud processing
        # For this sandbox, we'll just log what would be sent.
        cloud_sync_interval = self.edge_config["cloud_sync_interval_seconds"]
        telemetry_topic = self.edge_config["telemetry_topic_cloud"]

        while True:
            await asyncio.sleep(cloud_sync_interval)

            avg_speed = self.local_feature_store.get("avg_speed_kph")
            max_engine_temp = self.local_feature_store.get("max_engine_temp_celsius")
            current_gps = self.local_feature_store.get("current_gps")

            if avg_speed is None or max_engine_temp is None or current_gps is None:
                logger.debug(f"[{self.vehicle_id}] No data yet to sync to cloud.")
                continue

            cloud_telemetry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "driver_id": self.vehicle_id,  # Using vehicle_id as driver_id for simplicity
                "current_node_id": "sim_node_"
                + str(int(current_gps[0] * 100)),  # Mock node ID
                "latitude": current_gps[0],
                "longitude": current_gps[1],
                "speed_kph": round(avg_speed, 2),
                "engine_temp_celsius": round(max_engine_temp, 2),
            }
            logger.info(
                f"[{self.vehicle_id}] Syncing to cloud ({telemetry_topic}): {json.dumps(cloud_telemetry)}"
            )
            # In a real system: KafkaProducer.send(telemetry_topic, value=json.dumps(cloud_telemetry))

    async def run(self):
        if not self.edge_config["enabled"]:
            logger.info(f"Edge computing for {self.vehicle_id} is disabled.")
            return

        sensor_task = asyncio.create_task(self._simulate_sensor_data())
        sync_task = asyncio.create_task(self._sync_to_cloud_loop())

        try:
            await asyncio.gather(sensor_task, sync_task)
        except asyncio.CancelledError:
            logger.info(f"RealtimeIoTProcessor for {self.vehicle_id} cancelled.")
        except Exception as e:
            logger.error(
                f"Error in RealtimeIoTProcessor for {self.vehicle_id}: {e}",
                exc_info=True,
            )


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    edge_computing:
      enabled: true
      buffer_size_seconds: 10
      sampling_interval_ms: 100
      max_speed_threshold_kph: 60
      max_engine_temp_threshold_celsius: 100
      local_alert_debounce_seconds: 10
      cloud_sync_interval_seconds: 5
      telemetry_topic_cloud: dev_telemetry_stream
"""
            )

    async def main_edge():
        vehicle_processor = RealtimeIoTProcessor("truck_42", config_file)
        print("Starting RealtimeIoTProcessor for truck_42 for 30 seconds...")
        try:
            await asyncio.wait_for(vehicle_processor.run(), timeout=30)
        except asyncio.TimeoutError:
            print("\nRealtimeIoTProcessor demo timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nRealtimeIoTProcessor demo stopped by user.")

        print("\nFinal local features (if any):")
        print(
            f"  Avg Speed: {vehicle_processor.local_feature_store.get('avg_speed_kph'):.2f} kph"
        )
        print(
            f"  Max Engine Temp: {vehicle_processor.local_feature_store.get('max_engine_temp_celsius'):.2f} C"
        )
        print(
            f"  Current GPS: {vehicle_processor.local_feature_store.get('current_gps')}"
        )

    asyncio.run(main_edge())
