import datetime
import json
import yaml
import os
import logging
import random
import asyncio
from collections import deque
from typing import Dict, Any
import redis
import pandas as pd


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "driver_telemetry:driver_A",
            json.dumps(
                {
                    "speed_kph": 60,
                    "braking_events_5min": 2,
                    "lane_departure_events_5min": 0,
                    "traffic_density_score": 0.3,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        self.client.set(
            "driver_wellbeing:driver_A",
            json.dumps(
                {
                    "fatigue_score": 0.2,
                    "stress_score": 0.1,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        self.client.set(
            "traffic_prediction:current_route_segment",
            json.dumps(
                {"predicted_travel_time_factor": 1.0, "congestion_level": "low"}
            ),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def stream_features_to_df(
        self, feature_group: str, pattern: str = "*"
    ):  # For mock, just returns a static dict
        return pd.DataFrame(
            [{"driver_id": "driver_A", "fatigue_score": 0.2, "stress_score": 0.1}]
        )  # Dummy DataFrame


class MockDisplayController:
    def __init__(self, driver_id: str):
        self.driver_id = driver_id

    async def update_dashboard(self, new_ui_config: Dict[str, Any]):
        logging.info(f"[{self.driver_id}] Display updated: {json.dumps(new_ui_config)}")

    async def show_critical_alert(self, message: str):
        logging.critical(f"[{self.driver_id}] CRITICAL UI ALERT: {message}")


class MockDriverWellbeingMonitor:  # To calculate scores based on raw data
    def __init__(self, config_path, environment):
        self.monitor_config = {
            "fatigue_score_threshold": 0.7,
            "stress_score_threshold": 0.8,
        }
        self.driver_telemetry_history = {}  # Mock historical data
        self.driver_telemetry_history["driver_A"] = deque(
            [(datetime.datetime.utcnow(), {"avg_speed_kph": 60})], maxlen=10
        )

    def _update_telemetry_history(self, driver_id: str, telemetry_data: Dict[str, Any]):
        if driver_id not in self.driver_telemetry_history:
            self.driver_telemetry_history[driver_id] = deque(maxlen=10)
        self.driver_telemetry_history[driver_id].append(
            (datetime.datetime.utcnow(), telemetry_data)
        )

    def _calculate_fatigue_score(self, driver_id: str) -> float:
        return 0.2 + random.uniform(-0.1, 0.1)  # Mock

    def _calculate_stress_score(self, driver_id: str) -> float:
        return 0.1 + random.uniform(-0.05, 0.05)  # Mock


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitiveLoadManager:
    def __init__(
        self,
        driver_id: str,
        config_path="conf/environments/prod.yaml",
        environment="dev",
    ):
        self.driver_id = driver_id
        self.config = self._load_config(config_path)
        self.clm_config = self.config["environments"][environment][
            "cognitive_load_manager"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.display_controller = MockDisplayController(driver_id)
        self.wellbeing_monitor = MockDriverWellbeingMonitor(
            config_path, environment
        )  # To calculate fresh scores if needed

        self.last_load_level = "low"
        self.last_ui_update_time = datetime.datetime.utcnow()

        logger.info(f"CognitiveLoadManager for Driver {self.driver_id} initialized.")

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
    cognitive_load_manager:
      enabled: true
      update_interval_seconds: 2
      telemetry_history_window_seconds: 60
      load_thresholds:
        low_max: 0.3
        medium_max: 0.6
      load_factors:
        speed_kph_factor: 0.005
        braking_events_factor: 0.1
        lane_departure_factor: 0.3
        traffic_density_factor: 0.2
        fatigue_score_factor: 0.4
        stress_score_factor: 0.3
      ui_configs:
        low_load:
          layout: "full_details"
          map_zoom: "medium"
          info_panels: ["eta", "next_turn", "driver_vitals", "weather_forecast"]
          alert_priority: "normal"
        medium_load:
          layout: "simplified"
          map_zoom: "close"
          info_panels: ["eta", "next_turn", "driver_vitals"]
          alert_priority: "high"
        high_load:
          layout: "critical_only"
          map_zoom: "very_close"
          info_panels: ["next_turn"]
          alert_priority: "urgent"
          hide_non_critical: true
    driver_wellbeing_monitor:
      enabled: false
      history_window_minutes: 10
      fatigue_score_threshold: 0.7
      stress_score_threshold: 0.8
    traffic_prediction:
      enabled: false
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _calculate_cognitive_load(
        self,
        telemetry: Dict[str, Any],
        wellbeing_data: Dict[str, Any],
        traffic_data: Dict[str, Any],
    ) -> float:
        """
        Calculates a real-time cognitive load score for the driver.
        """
        load_score = 0.0

        # Telemetry-based factors
        speed = telemetry.get("speed_kph", 0)
        braking_events = telemetry.get("braking_events_5min", 0)
        lane_departure = telemetry.get("lane_departure_events_5min", 0)
        traffic_density = telemetry.get(
            "traffic_density_score", 0
        )  # Placeholder for traffic complexity from map

        load_score += (
            speed * self.clm_config["load_factors"]["speed_kph_factor"]
        )  # Higher speed, more load
        load_score += (
            braking_events * self.clm_config["load_factors"]["braking_events_factor"]
        )  # Sudden braking, higher load
        load_score += (
            lane_departure * self.clm_config["load_factors"]["lane_departure_factor"]
        )  # Errors, very high load
        load_score += (
            traffic_density * self.clm_config["load_factors"]["traffic_density_factor"]
        )  # Denser traffic, more load

        # Wellbeing factors
        fatigue_score = wellbeing_data.get("fatigue_score", 0)
        stress_score = wellbeing_data.get("stress_score", 0)
        load_score += (
            fatigue_score * self.clm_config["load_factors"]["fatigue_score_factor"]
        )
        load_score += (
            stress_score * self.clm_config["load_factors"]["stress_score_factor"]
        )

        # Traffic prediction factors (e.g., unexpected congestion)
        congestion_level = traffic_data.get("congestion_level", "low")
        if congestion_level == "medium":
            load_score += 0.1
        elif congestion_level == "high":
            load_score += 0.2

        return min(max(0, load_score), 1.0)  # Cap between 0 and 1

    def _get_ui_config_for_load(self, cognitive_load: float) -> Dict[str, Any]:
        """Maps cognitive load score to a predefined UI configuration."""
        if cognitive_load > self.clm_config["load_thresholds"]["medium_max"]:
            self.last_load_level = "high"
            return self.clm_config["ui_configs"]["high_load"]
        elif cognitive_load > self.clm_config["load_thresholds"]["low_max"]:
            self.last_load_level = "medium"
            return self.clm_config["ui_configs"]["medium_load"]
        else:
            self.last_load_level = "low"
            return self.clm_config["ui_configs"]["low_load"]

    async def run_cognitive_load_management_loop(self):
        """Main loop for continuously assessing load and updating UI."""
        if not self.clm_config["enabled"]:
            logger.info(f"Cognitive load manager for {self.driver_id} is disabled.")
            return

        while True:
            try:
                # 1. Fetch real-time data
                telemetry = (
                    self.feature_store.get_feature("driver_telemetry", self.driver_id)
                    or {}
                )
                wellbeing_data = (
                    self.feature_store.get_feature("driver_wellbeing", self.driver_id)
                    or {}
                )
                traffic_data = (
                    self.feature_store.get_feature(
                        "traffic_prediction", "current_route_segment"
                    )
                    or {}
                )  # Mock for current segment

                if not telemetry:
                    logger.warning(
                        f"No telemetry for {self.driver_id}. Cannot calculate cognitive load."
                    )
                    await asyncio.sleep(self.clm_config["update_interval_seconds"])
                    continue

                # For demo, update wellbeing monitor with raw telemetry to get 'fresh' scores
                self.wellbeing_monitor._update_telemetry_history(
                    self.driver_id, telemetry
                )
                wellbeing_data["fatigue_score"] = (
                    self.wellbeing_monitor._calculate_fatigue_score(self.driver_id)
                )
                wellbeing_data["stress_score"] = (
                    self.wellbeing_monitor._calculate_stress_score(self.driver_id)
                )

                # 2. Calculate cognitive load
                cognitive_load = self._calculate_cognitive_load(
                    telemetry, wellbeing_data, traffic_data
                )
                logger.debug(
                    f"Driver {self.driver_id} cognitive load: {cognitive_load:.2f} ({self.last_load_level})"
                )

                # 3. Determine and apply UI configuration
                new_ui_config = self._get_ui_config_for_load(cognitive_load)

                # Only update UI if there's a significant change in config or a long time has passed
                if (
                    new_ui_config != self.display_controller.last_config
                    or (
                        datetime.datetime.utcnow() - self.last_ui_update_time
                    ).total_seconds()
                    > self.clm_config["update_interval_seconds"] * 5
                ):
                    await self.display_controller.update_dashboard(new_ui_config)
                    self.display_controller.last_config = (
                        new_ui_config  # Mock for tracking
                    )
                    self.last_ui_update_time = datetime.datetime.utcnow()

            except Exception as e:
                logger.error(
                    f"Error in cognitive load management loop for {self.driver_id}: {e}",
                    exc_info=True,
                )

            await asyncio.sleep(self.clm_config["update_interval_seconds"])


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
    cognitive_load_manager:
      enabled: true
      update_interval_seconds: 2
      telemetry_history_window_seconds: 60
      load_thresholds:
        low_max: 0.3
        medium_max: 0.6
      load_factors:
        speed_kph_factor: 0.005
        braking_events_factor: 0.1
        lane_departure_factor: 0.3
        traffic_density_factor: 0.2
        fatigue_score_factor: 0.4
        stress_score_factor: 0.3
      ui_configs:
        low_load:
          layout: "full_details"
          map_zoom: "medium"
          info_panels: ["eta", "next_turn", "driver_vitals", "weather_forecast"]
          alert_priority: "normal"
        medium_load:
          layout: "simplified"
          map_zoom: "close"
          info_panels: ["eta", "next_turn", "driver_vitals"]
          alert_priority: "high"
        high_load:
          layout: "critical_only"
          map_zoom: "very_close"
          info_panels: ["next_turn"]
          alert_priority: "urgent"
          hide_non_critical: true
    driver_wellbeing_monitor:
      enabled: false
      history_window_minutes: 10
      fatigue_score_threshold: 0.7
      stress_score_threshold: 0.8
    traffic_prediction:
      enabled: false
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis. Populating dummy data for CognitiveLoadManager.")
        # Initial low load
        r.set(
            "driver_telemetry:driver_A",
            json.dumps(
                {
                    "speed_kph": 30,
                    "braking_events_5min": 0,
                    "lane_departure_events_5min": 0,
                    "traffic_density_score": 0.1,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        r.set(
            "driver_wellbeing:driver_A",
            json.dumps(
                {
                    "fatigue_score": 0.1,
                    "stress_score": 0.05,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        r.set(
            "traffic_prediction:current_route_segment",
            json.dumps(
                {"predicted_travel_time_factor": 1.0, "congestion_level": "low"}
            ),
        )
    except redis.exceptions.ConnectionError:
        print("Redis not running. Cognitive load manager will start with empty data.")

    async def simulate_dynamic_telemetry(r: redis.StrictRedis, driver_id: str):
        # Simulate load increasing over time
        for i in range(15):
            current_speed = 30 + i * 5  # Increases from 30 to 100
            braking_events = max(0, i // 5)  # 0, 0, 0, 1, 1, 1, 2, 2, 2
            traffic_density = 0.1 + i * 0.05
            fatigue = 0.1 + i * 0.03
            stress = 0.05 + i * 0.04
            congestion = "low"
            if i > 7:
                congestion = "medium"
            if i > 12:
                congestion = "high"

            r.set(
                f"driver_telemetry:{driver_id}",
                json.dumps(
                    {
                        "speed_kph": current_speed,
                        "braking_events_5min": braking_events,
                        "lane_departure_events_5min": 0,
                        "traffic_density_score": traffic_density,
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    }
                ),
            )
            r.set(
                f"driver_wellbeing:{driver_id}",
                json.dumps(
                    {
                        "fatigue_score": fatigue,
                        "stress_score": stress,
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    }
                ),
            )
            r.set(
                "traffic_prediction:current_route_segment",
                json.dumps(
                    {
                        "predicted_travel_time_factor": 1.0 + traffic_density,
                        "congestion_level": congestion,
                    }
                ),
            )
            await asyncio.sleep(
                2
            )  # Update every 2 seconds for a total of 30 seconds simulation

    async def main_cognitive_load_manager():
        driver_id = "driver_A"
        cl_manager = CognitiveLoadManager(driver_id, config_file)

        asyncio.create_task(
            simulate_dynamic_telemetry(
                redis.StrictRedis(
                    host="localhost", port=6379, db=0, decode_responses=True
                ),
                driver_id,
            )
        )

        print(f"Starting CognitiveLoadManager for {driver_id} for 35 seconds...")
        try:
            await asyncio.wait_for(
                cl_manager.run_cognitive_load_management_loop(), timeout=35
            )
        except asyncio.TimeoutError:
            print(
                f"\nCognitiveLoadManager demo for {driver_id} timed out after 35 seconds."
            )
        except KeyboardInterrupt:
            print(f"\nCognitiveLoadManager demo for {driver_id} stopped by user.")

    asyncio.run(main_cognitive_load_manager())
