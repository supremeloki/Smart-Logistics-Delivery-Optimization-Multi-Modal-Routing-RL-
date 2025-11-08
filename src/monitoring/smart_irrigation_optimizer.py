import datetime
import json
import yaml
import os
import logging
import asyncio
from typing import Dict, Any, Tuple
import redis
import numpy as np


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "soil_sensor:ZONE_A_S1",
            json.dumps(
                {"moisture_percent": 35.0, "temp_c": 22.5, "nutrient_level": 0.7}
            ),
        )
        self.client.set(
            "weather_forecast:ZONE_A",
            json.dumps(
                {
                    "precipitation_mm_24h": 0.5,
                    "temp_c_avg": 23.0,
                    "humidity_percent": 60,
                }
            ),
        )
        self.client.set(
            "plant_profile:WHEAT_FIELD_1",
            json.dumps(
                {
                    "min_moisture_percent": 30,
                    "max_moisture_percent": 70,
                    "optimal_moisture": 55,
                    "growth_stage": "flowering",
                    "evapotranspiration_rate": 0.8,
                }
            ),
        )
        self.client.set(
            "irrigation_zone:ZONE_A",
            json.dumps(
                {
                    "valve_status": "closed",
                    "current_flow_lps": 0,
                    "last_irrigation_end": (
                        datetime.datetime.utcnow() - datetime.timedelta(hours=24)
                    ).isoformat()
                    + "Z",
                }
            ),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def get_all_features_by_group(self, feature_group: str):
        keys = self.client.keys(f"{feature_group}:*")
        return (
            {k.split(":")[1]: json.loads(self.client.get(k)) for k in keys}
            if keys
            else {}
        )

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))


class MockIrrigationControllerAPI:
    def __init__(self, zone_id: str):
        self.zone_id = zone_id

    async def open_valve(self, duration_minutes: int, flow_rate_lps: float):
        logging.info(
            f"[{self.zone_id}] Simulating opening valve for {duration_minutes} min at {flow_rate_lps} L/s."
        )
        await asyncio.sleep(duration_minutes * 60 / 10)  # Faster simulation
        return {"status": "SUCCESS", "message": "Valve opened."}

    async def close_valve(self):
        logging.info(f"[{self.zone_id}] Simulating closing valve.")
        await asyncio.sleep(0.5)
        return {"status": "SUCCESS", "message": "Valve closed."}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartIrrigationOptimizer:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.irrigation_config = self.config["environments"][environment][
            "smart_irrigation_optimizer"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.irrigation_controllers = {
            zone: MockIrrigationControllerAPI(zone)
            for zone in self.irrigation_config["managed_zones"]
        }
        self.active_irrigation = {}  # zone_id -> task_object

        logger.info("SmartIrrigationOptimizer initialized.")

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
    smart_irrigation_optimizer:
      enabled: true
      optimization_interval_seconds: 10
      managed_zones: ["ZONE_A", "ZONE_B"]
      irrigation_threshold_buffer: 5 # % below optimal moisture to trigger
      min_irrigation_duration_minutes: 10
      max_irrigation_duration_minutes: 60
      default_flow_rate_lps: 0.5
      soil_capacity_mm_per_percent_moisture: 0.5 # Example: 0.5 mm water per % moisture
      weather_impact_factor: 0.8 # How much weather forecast reduces needed irrigation
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    async def _determine_irrigation_needs(
        self, zone_id: str, plant_profile_id: str
    ) -> Tuple[bool, int, float, str]:
        """Determines if and how much irrigation is needed for a zone."""
        soil_sensors = self.feature_store.get_all_features_by_group(
            f"soil_sensor:{zone_id}_*"
        )  # All sensors in zone
        if not soil_sensors:
            return False, 0, 0.0, "No soil sensor data."

        avg_moisture = np.mean([s["moisture_percent"] for s in soil_sensors.values()])

        plant_profile = self.feature_store.get_feature(
            "plant_profile", plant_profile_id
        )
        if not plant_profile:
            return False, 0, 0.0, f"No plant profile found for {plant_profile_id}."

        optimal_moisture = plant_profile.get("optimal_moisture", 50)
        plant_profile.get("min_moisture_percent", 30)

        # Consider weather forecast
        weather_forecast = self.feature_store.get_feature("weather_forecast", zone_id)
        expected_precipitation_mm = weather_forecast.get("precipitation_mm_24h", 0)

        # Calculate target moisture, adjusted for buffer
        target_moisture = (
            optimal_moisture - self.irrigation_config["irrigation_threshold_buffer"]
        )

        if avg_moisture < target_moisture:
            moisture_deficit_percent = target_moisture - avg_moisture

            # Convert moisture deficit to required water (mm)
            water_needed_mm = (
                moisture_deficit_percent
                * self.irrigation_config["soil_capacity_mm_per_percent_moisture"]
            )

            # Adjust water needed based on forecast precipitation
            net_water_needed_mm = max(
                0,
                water_needed_mm
                - (
                    expected_precipitation_mm
                    * self.irrigation_config["weather_impact_factor"]
                ),
            )

            if net_water_needed_mm > 0:
                # Assuming 1 L/s for 1 minute = X mm of water over a certain area.
                # Simplified: calculate duration based on flow rate and required water
                # Example: If 0.5 L/s = 30 L/min. Need to map to area covered by zone.
                # For demo, let's assume 1 L/s for 1 min adds 0.1 mm to the entire zone.
                # This needs a detailed zone area in a real system.
                estimated_duration_minutes = int(
                    net_water_needed_mm
                    * 10
                    / self.irrigation_config["default_flow_rate_lps"]
                )

                # Clamp duration to min/max
                duration = np.clip(
                    estimated_duration_minutes,
                    self.irrigation_config["min_irrigation_duration_minutes"],
                    self.irrigation_config["max_irrigation_duration_minutes"],
                )

                return (
                    True,
                    int(duration),
                    self.irrigation_config["default_flow_rate_lps"],
                    "Irrigation needed.",
                )

        return (
            False,
            0,
            0.0,
            "Moisture levels sufficient or expected precipitation covers deficit.",
        )

    async def _manage_irrigation_zone(self, zone_id: str, plant_profile_id: str):
        """Manages the irrigation for a single zone."""
        zone_status = self.feature_store.get_feature("irrigation_zone", zone_id) or {}
        is_valve_open = zone_status.get("valve_status") == "open"

        irrigate, duration, flow_rate, reason = await self._determine_irrigation_needs(
            zone_id, plant_profile_id
        )

        if irrigate and not is_valve_open and zone_id not in self.active_irrigation:
            logger.info(
                f"Triggering irrigation for {zone_id}: {reason} for {duration} min at {flow_rate} L/s."
            )
            irrigation_task = asyncio.create_task(
                self.irrigation_controllers[zone_id].open_valve(duration, flow_rate)
            )
            self.active_irrigation[zone_id] = irrigation_task  # Store task to track

            zone_status["valve_status"] = "open"
            zone_status["current_flow_lps"] = flow_rate
            zone_status["last_irrigation_start"] = (
                datetime.datetime.utcnow().isoformat() + "Z"
            )
            self.feature_store.set_feature("irrigation_zone", zone_id, zone_status)

            # Schedule valve closure
            await asyncio.sleep(duration * 60 / 10)  # Wait for simulated duration
            await self.irrigation_controllers[zone_id].close_valve()

            zone_status["valve_status"] = "closed"
            zone_status["current_flow_lps"] = 0
            zone_status["last_irrigation_end"] = (
                datetime.datetime.utcnow().isoformat() + "Z"
            )
            self.feature_store.set_feature("irrigation_zone", zone_id, zone_status)
            del self.active_irrigation[zone_id]
            logger.info(f"Irrigation completed for {zone_id}.")

        elif not irrigate and is_valve_open:
            logger.info(f"Stopping irrigation for {zone_id}: {reason}.")
            await self.irrigation_controllers[zone_id].close_valve()
            zone_status["valve_status"] = "closed"
            zone_status["current_flow_lps"] = 0
            zone_status["last_irrigation_end"] = (
                datetime.datetime.utcnow().isoformat() + "Z"
            )
            self.feature_store.set_feature("irrigation_zone", zone_id, zone_status)
            if zone_id in self.active_irrigation:
                self.active_irrigation[zone_id].cancel()
                del self.active_irrigation[zone_id]
        else:
            logger.debug(f"Zone {zone_id}: No action needed. Reason: {reason}.")

    async def run_optimizer_loop(self):
        """Main loop for continuously optimizing irrigation across all managed zones."""
        if not self.irrigation_config["enabled"]:
            logger.info("Smart Irrigation Optimizer is disabled.")
            return

        while True:
            logger.info("Running smart irrigation optimization cycle...")

            # Map zones to plant profiles (simplified for demo)
            zone_plant_map = {"ZONE_A": "WHEAT_FIELD_1", "ZONE_B": "CORN_FIELD_1"}

            tasks = []
            for zone_id in self.irrigation_config["managed_zones"]:
                plant_profile_id = zone_plant_map.get(
                    zone_id, "DEFAULT_PLANT_PROFILE"
                )  # Fallback
                tasks.append(self._manage_irrigation_zone(zone_id, plant_profile_id))

            await asyncio.gather(*tasks)  # Run management for all zones concurrently

            await asyncio.sleep(self.irrigation_config["optimization_interval_seconds"])


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
    smart_irrigation_optimizer:
      enabled: true
      optimization_interval_seconds: 5 # Faster for demo
      managed_zones: ["ZONE_A", "ZONE_B"]
      irrigation_threshold_buffer: 5
      min_irrigation_duration_minutes: 5
      max_irrigation_duration_minutes: 30
      default_flow_rate_lps: 0.8
      soil_capacity_mm_per_percent_moisture: 0.5
      weather_impact_factor: 0.7
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis. Populating dummy data for SmartIrrigationOptimizer.")
        r.set(
            "soil_sensor:ZONE_A_S1",
            json.dumps(
                {"moisture_percent": 32.0, "temp_c": 22.5, "nutrient_level": 0.7}
            ),
        )  # Needs irrigation
        r.set(
            "soil_sensor:ZONE_B_S1",
            json.dumps(
                {"moisture_percent": 60.0, "temp_c": 21.0, "nutrient_level": 0.8}
            ),
        )  # No irrigation needed
        r.set(
            "weather_forecast:ZONE_A",
            json.dumps(
                {
                    "precipitation_mm_24h": 0.0,
                    "temp_c_avg": 25.0,
                    "humidity_percent": 50,
                }
            ),
        )
        r.set(
            "weather_forecast:ZONE_B",
            json.dumps(
                {
                    "precipitation_mm_24h": 5.0,
                    "temp_c_avg": 20.0,
                    "humidity_percent": 70,
                }
            ),
        )
        r.set(
            "plant_profile:WHEAT_FIELD_1",
            json.dumps(
                {
                    "min_moisture_percent": 30,
                    "max_moisture_percent": 70,
                    "optimal_moisture": 55,
                    "growth_stage": "flowering",
                    "evapotranspiration_rate": 0.8,
                }
            ),
        )
        r.set(
            "plant_profile:CORN_FIELD_1",
            json.dumps(
                {
                    "min_moisture_percent": 40,
                    "max_moisture_percent": 80,
                    "optimal_moisture": 65,
                    "growth_stage": "vegetative",
                    "evapotranspiration_rate": 0.9,
                }
            ),
        )
        r.set(
            "irrigation_zone:ZONE_A",
            json.dumps(
                {
                    "valve_status": "closed",
                    "current_flow_lps": 0,
                    "last_irrigation_end": (
                        datetime.datetime.utcnow() - datetime.timedelta(hours=24)
                    ).isoformat()
                    + "Z",
                }
            ),
        )
        r.set(
            "irrigation_zone:ZONE_B",
            json.dumps(
                {
                    "valve_status": "closed",
                    "current_flow_lps": 0,
                    "last_irrigation_end": (
                        datetime.datetime.utcnow() - datetime.timedelta(hours=12)
                    ).isoformat()
                    + "Z",
                }
            ),
        )

    except redis.exceptions.ConnectionError:
        print(
            "Redis not running. Smart Irrigation Optimizer will start with empty data."
        )

    async def main_irrigation_optimizer():
        optimizer = SmartIrrigationOptimizer(config_file)
        print("Starting SmartIrrigationOptimizer for 30 seconds...")
        try:
            await asyncio.wait_for(optimizer.run_optimizer_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("\nSmartIrrigationOptimizer demo timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nSmartIrrigationOptimizer demo stopped by user.")

    asyncio.run(main_irrigation_optimizer())
