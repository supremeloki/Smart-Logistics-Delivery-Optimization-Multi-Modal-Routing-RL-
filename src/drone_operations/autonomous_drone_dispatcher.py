import datetime
import json
import numpy as np
import yaml
import os
import logging
import random
import asyncio
from typing import Dict, Any, List, Tuple


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "package:PKG_001",
            json.dumps(
                {
                    "weight_kg": 0.8,
                    "dimensions_cm": [20, 15, 5],
                    "origin_lat": 35.7,
                    "origin_lon": 51.4,
                    "destination_lat": 35.71,
                    "destination_lon": 51.41,
                    "urgency_score": 0.9,
                    "status": "ready_for_dispatch",
                }
            ),
        )
        self.client.set(
            "drone:DRN_001",
            json.dumps(
                {
                    "status": "idle",
                    "current_lat": 35.7,
                    "current_lon": 51.4,
                    "battery_percent": 95,
                    "max_payload_kg": 1.5,
                    "max_range_km": 10,
                    "last_maintenance": (
                        datetime.datetime.utcnow() - datetime.timedelta(days=7)
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


class MockWeatherImpactAssessor:
    def __init__(self, config_path, environment):
        self.config_path = config_path
        self.environment = environment

    def assess_impact_on_route_segment(
        self,
        lat: float,
        lon: float,
        current_speed_kph: float,
        segment_free_flow_time_sec: float,
        segment_length_m: float,
    ) -> dict:
        condition = random.choice(["clear", "light_rain", "windy"])
        return {
            "adjusted_travel_time_sec": segment_free_flow_time_sec
            * (1.0 + random.uniform(0, 0.2)),
            "safety_score": 1.0 - (0.1 if condition != "clear" else 0.0),
            "weather_condition": condition,
            "recommendation": (
                "Normal operation."
                if condition == "clear"
                else "Caution due to weather."
            ),
        }

    def get_route_weather_summary(
        self, path_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        conditions = [
            self.assess_impact_on_route_segment(
                s["lat"], s["lon"], 0, s["free_flow_time_sec"], s["length_m"]
            )["weather_condition"]
            for s in path_segments
        ]
        dominant = max(set(conditions), key=conditions.count)
        return {
            "overall_dominant_weather": dominant,
            "overall_safety_score": 1.0 if dominant == "clear" else 0.8,
        }


class MockGISService:
    def __init__(self):
        pass

    def calculate_straight_line_distance(self, lat1, lon1, lat2, lon2):
        return (
            np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111.0 * 1000
        )  # Rough meters

    def estimate_drone_route_segments(self, start_lat, start_lon, end_lat, end_lon):
        dist = self.calculate_straight_line_distance(
            start_lat, start_lon, end_lat, end_lon
        )
        segments = []
        for i in range(int(dist / 500) + 1):  # Every 500m
            progress = i * 500 / dist if dist > 0 else 0
            lat = start_lat + (end_lat - start_lat) * progress
            lon = start_lon + (end_lon - start_lon) * progress
            segments.append(
                {"lat": lat, "lon": lon, "free_flow_time_sec": 30, "length_m": 500}
            )
        return segments


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutonomousDroneDispatcher:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.drone_config = self.config["environments"][environment]["drone_dispatcher"]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.weather_assessor = MockWeatherImpactAssessor(config_path, environment)
        self.gis_service = MockGISService()

        self.drone_task_queue = asyncio.Queue()
        self.active_dispatches = {}  # Track ongoing dispatches

        logger.info("AutonomousDroneDispatcher initialized.")

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
    drone_dispatcher:
      enabled: true
      dispatch_interval_seconds: 5
      min_drone_battery_percent: 30
      max_drone_range_safety_factor: 0.8 # Use 80% of max range
      weather_safety_score_threshold: 0.7
      max_payload_safety_factor: 0.9
      no_fly_zones: [] # Example: [[lat1, lon1, lat2, lon2], ...]
      drone_base_stations: # Example: { "base_1": {"lat": 35.7, "lon": 51.4}, ... }
        base_1: {"lat": 35.7, "lon": 51.4}
        base_2: {"lat": 35.75, "lon": 51.5}
    weather_impact_assessor: # Dummy for mock
      enabled: false
      api_key: MOCK
    alerting: # Dummy for mock
      slack:
        enabled: false
      email:
        enabled: false
      rules: []
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _is_within_no_fly_zone(self, lat, lon):
        for zone in self.drone_config["no_fly_zones"]:
            if (
                zone[0] <= lat <= zone[2] and zone[1] <= lon <= zone[3]
            ):  # Simple bounding box check
                return True
        return False

    async def _evaluate_dispatch_feasibility(
        self, package: Dict[str, Any], drone: Dict[str, Any]
    ) -> Tuple[bool, str, float, float]:
        """Evaluates if a package can be dispatched by a given drone."""
        package_origin = (package["origin_lat"], package["origin_lon"])
        package_destination = (package["destination_lat"], package["destination_lon"])
        drone_current_location = (drone["current_lat"], drone["current_lon"])

        # 1. Payload capacity
        if (
            package["weight_kg"]
            > drone["max_payload_kg"] * self.drone_config["max_payload_safety_factor"]
        ):
            return False, "Payload too heavy", 0.0, 0.0

        # 2. Battery and Range
        distance_to_pickup = (
            self.gis_service.calculate_straight_line_distance(
                drone_current_location[0],
                drone_current_location[1],
                package_origin[0],
                package_origin[1],
            )
            / 1000
        )  # km

        distance_delivery = (
            self.gis_service.calculate_straight_line_distance(
                package_origin[0],
                package_origin[1],
                package_destination[0],
                package_destination[1],
            )
            / 1000
        )  # km

        # Assume return to nearest base station after delivery
        closest_base_dist = np.inf
        for base_id, base_loc in self.drone_config["drone_base_stations"].items():
            dist = (
                self.gis_service.calculate_straight_line_distance(
                    package_destination[0],
                    package_destination[1],
                    base_loc["lat"],
                    base_loc["lon"],
                )
                / 1000
            )
            closest_base_dist = min(closest_base_dist, dist)

        total_trip_distance = distance_to_pickup + distance_delivery + closest_base_dist

        if (
            total_trip_distance
            > drone["max_range_km"] * self.drone_config["max_drone_range_safety_factor"]
        ):
            return (
                False,
                f"Trip ({total_trip_distance:.1f}km) exceeds drone range ({drone['max_range_km'] * self.drone_config['max_drone_range_safety_factor']:.1f}km)",
                0.0,
                0.0,
            )

        # Mock calculation: 10% battery per 1km (simplified)
        estimated_battery_drain = total_trip_distance * (
            100 / (drone["max_range_km"] * 0.8)
        )  # Percentage
        if (
            drone["battery_percent"] - estimated_battery_drain
            < self.drone_config["min_drone_battery_percent"]
        ):
            return (
                False,
                f"Insufficient battery ({drone['battery_percent']:.1f}% -> {drone['battery_percent'] - estimated_battery_drain:.1f}%)",
                0.0,
                0.0,
            )

        # 3. Weather Conditions
        # Estimate route segments for weather assessment (simplified to just one segment)
        route_segments = self.gis_service.estimate_drone_route_segments(
            package_origin[0],
            package_origin[1],
            package_destination[0],
            package_destination[1],
        )
        weather_summary = self.weather_assessor.get_route_weather_summary(
            route_segments
        )
        if (
            weather_summary["overall_safety_score"]
            < self.drone_config["weather_safety_score_threshold"]
        ):
            return (
                False,
                f"Unfavorable weather: {weather_summary['overall_dominant_weather']}",
                0.0,
                0.0,
            )

        # 4. No-fly zones
        if self._is_within_no_fly_zone(
            package_origin[0], package_origin[1]
        ) or self._is_within_no_fly_zone(
            package_destination[0], package_destination[1]
        ):
            return False, "Route intersects no-fly zone", 0.0, 0.0

        # Calculate a "dispatch score" for prioritization (e.g., package urgency, drone availability)
        dispatch_score = (
            package["urgency_score"] * 0.7
            + (1 - (drone["battery_percent"] / 100)) * 0.1
            + (1 / (distance_delivery + 1)) * 0.2
        )
        estimated_travel_time = (
            sum(s["free_flow_time_sec"] for s in route_segments)
            * (1.0 / weather_summary["overall_safety_score"])
            / 60
        )  # Minutes

        return True, "Feasible", dispatch_score, estimated_travel_time

    async def _assign_and_dispatch_drone(
        self, package_id: str, drone_id: str, estimated_delivery_time_min: float
    ):
        """Simulates assigning a package to a drone and dispatching it."""
        logger.info(
            f"Dispatching Drone {drone_id} for Package {package_id}. ETA: {estimated_delivery_time_min:.1f} min."
        )
        self.active_dispatches[package_id] = {
            "drone_id": drone_id,
            "dispatch_time": datetime.datetime.utcnow(),
            "estimated_delivery_time": datetime.datetime.utcnow()
            + datetime.timedelta(minutes=estimated_delivery_time_min),
            "status": "en_route",
        }
        # Update Feature Store (mock): change drone status to 'en_route', package status to 'dispatched'
        drone_info = self.feature_store.get_feature("drone", drone_id)
        if drone_info:
            drone_info["status"] = "en_route"
            self.feature_store.client.set(f"drone:{drone_id}", json.dumps(drone_info))

        package_info = self.feature_store.get_feature("package", package_id)
        if package_info:
            package_info["status"] = "dispatched"
            self.feature_store.client.set(
                f"package:{package_id}", json.dumps(package_info)
            )

        # Simulate drone flight
        await asyncio.sleep(
            estimated_delivery_time_min * 60
        )  # Block for simulated delivery time

        logger.info(f"Drone {drone_id} delivered Package {package_id}.")
        # Update Feature Store: change drone status to 'idle' (or 'returning_to_base'), package status to 'delivered'
        drone_info = self.feature_store.get_feature("drone", drone_id)
        if drone_info:
            drone_info["status"] = "idle"
            estimated_battery_drain_from_total_trip = 10.0  # Dummy value for demo
            drone_info["battery_percent"] -= estimated_battery_drain_from_total_trip
            self.feature_store.client.set(
                f"drone:{drone_id}", json.dumps(drone_info)
            )  # Update battery
        package_info = self.feature_store.get_feature("package", package_id)
        if package_info:
            package_info["status"] = "delivered"
            package_info["actual_delivery_time"] = (
                datetime.datetime.utcnow().isoformat() + "Z"
            )
            self.feature_store.client.set(
                f"package:{package_id}", json.dumps(package_info)
            )

        del self.active_dispatches[package_id]  # Remove from active list

    async def dispatch_loop(self):
        """Main loop for finding packages and assigning drones."""
        if not self.drone_config["enabled"]:
            logger.info("Drone dispatcher is disabled.")
            return

        while True:
            logger.info("Searching for packages to dispatch...")
            ready_packages = self.feature_store.get_all_features_by_group("package")
            ready_packages = {
                pid: p
                for pid, p in ready_packages.items()
                if p.get("status") == "ready_for_dispatch"
            }

            idle_drones = self.feature_store.get_all_features_by_group("drone")
            idle_drones = {
                did: d for did, d in idle_drones.items() if d.get("status") == "idle"
            }

            if not ready_packages or not idle_drones:
                logger.info("No ready packages or idle drones. Waiting...")
                await asyncio.sleep(self.drone_config["dispatch_interval_seconds"])
                continue

            best_assignment_score = -1
            best_package_id = None
            best_drone_id = None
            estimated_delivery_time = 0.0

            for package_id, package_data in ready_packages.items():
                for drone_id, drone_data in idle_drones.items():
                    is_feasible, reason, score, eta_min = (
                        await self._evaluate_dispatch_feasibility(
                            package_data, drone_data
                        )
                    )
                    if is_feasible and score > best_assignment_score:
                        best_assignment_score = score
                        best_package_id = package_id
                        best_drone_id = drone_id
                        estimated_delivery_time = eta_min

            if best_package_id and best_drone_id:
                await self._assign_and_dispatch_drone(
                    best_package_id, best_drone_id, estimated_delivery_time
                )
            else:
                logger.info(
                    "No optimal drone assignment found for current packages/drones."
                )

            await asyncio.sleep(self.drone_config["dispatch_interval_seconds"])


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
    drone_dispatcher:
      enabled: true
      dispatch_interval_seconds: 5
      min_drone_battery_percent: 30
      max_drone_range_safety_factor: 0.8
      weather_safety_score_threshold: 0.7
      max_payload_safety_factor: 0.9
      no_fly_zones: []
      drone_base_stations:
        base_1: {"lat": 35.7, "lon": 51.4}
        base_2: {"lat": 35.75, "lon": 51.5}
    weather_impact_assessor:
      enabled: true
      api_key: MOCK
    alerting:
      slack:
        enabled: false
      email:
        enabled: false
      rules: []
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis. Populating dummy data for Drone Dispatcher.")
        # Create a package ready for dispatch
        r.set(
            "package:PKG_001",
            json.dumps(
                {
                    "weight_kg": 0.5,
                    "dimensions_cm": [20, 15, 5],
                    "origin_lat": 35.705,
                    "origin_lon": 51.405,
                    "destination_lat": 35.715,
                    "destination_lon": 51.415,
                    "urgency_score": 0.9,
                    "status": "ready_for_dispatch",
                }
            ),
        )
        r.set(
            "drone:DRN_001",
            json.dumps(
                {
                    "status": "idle",
                    "current_lat": 35.7,
                    "current_lon": 51.4,
                    "battery_percent": 90,
                    "max_payload_kg": 1.5,
                    "max_range_km": 10,
                    "last_maintenance": (
                        datetime.datetime.utcnow() - datetime.timedelta(days=7)
                    ).isoformat()
                    + "Z",
                }
            ),
        )
        r.set(
            "drone:DRN_002",
            json.dumps(
                {
                    "status": "idle",
                    "current_lat": 35.76,
                    "current_lon": 51.51,
                    "battery_percent": 25,  # Low battery
                    "max_payload_kg": 2.0,
                    "max_range_km": 12,
                    "last_maintenance": (
                        datetime.datetime.utcnow() - datetime.timedelta(days=10)
                    ).isoformat()
                    + "Z",
                }
            ),
        )
        # Add a package that's too heavy
        r.set(
            "package:PKG_002",
            json.dumps(
                {
                    "weight_kg": 1.8,
                    "dimensions_cm": [30, 20, 10],
                    "origin_lat": 35.72,
                    "origin_lon": 51.42,
                    "destination_lat": 35.73,
                    "destination_lon": 51.43,
                    "urgency_score": 0.7,
                    "status": "ready_for_dispatch",
                }
            ),
        )
    except redis.exceptions.ConnectionError:
        print("Redis not running. Drone dispatcher will start with empty data.")

    async def main_drone_dispatcher():
        dispatcher = AutonomousDroneDispatcher(config_file)
        print("Starting AutonomousDroneDispatcher for 30 seconds...")
        try:
            await asyncio.wait_for(dispatcher.dispatch_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("\nAutonomousDroneDispatcher demo timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nAutonomousDroneDispatcher demo stopped by user.")

    asyncio.run(main_drone_dispatcher())
