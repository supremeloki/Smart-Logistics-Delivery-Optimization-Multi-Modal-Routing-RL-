import datetime
import json
import yaml
import os
import logging
import random
import asyncio
from typing import Dict, Any
import networkx as nx
import redis

from feature_forge.feature_store_client import FeatureStoreClient
from driver_management.driver_wellbeing_monitor import (
    DriverWellbeingMonitor,
)  # To get wellbeing scores
from routing.osmnx_processor import OSMNxProcessor  # For map context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AugmentedRealityDashboard:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.ar_config = self.config["environments"][environment]["ar_dashboard"]

        self.feature_store_client = FeatureStoreClient(config_path, environment)
        self.osm_processor = OSMNxProcessor(
            self.config["environments"][environment].get(
                "osm_processing_config_path", "conf/osm_processing_config.yaml"
            )
        )
        self.graph = self.osm_processor.get_graph()  # Static map data

        # Integration with other services to fetch data
        self.wellbeing_monitor = DriverWellbeingMonitor(config_path, environment)
        # self.maintenance_service = PredictiveMaintenanceService(
        #     config_path, environment
        # )

        logger.info("AugmentedRealityDashboard backend initialized.")

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
    ar_dashboard:
      enabled: true
      update_interval_seconds: 2
      map_tile_server_url: https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
      realtime_overlay_refresh_rate_ms: 500
      max_alert_overlays: 3
    osm_processing_config_path: conf/osm_processing_config.yaml
    driver_wellbeing_monitor: # Dummy config for DriverWellbeingMonitor to initialize
      enabled: false 
      history_window_minutes: 10
      fatigue_score_threshold: 0.7
      stress_score_threshold: 0.8
    predictive_maintenance: # Dummy config for PredictiveMaintenanceService to initialize
      enabled: false
      history_window_minutes: 10
      model_dir: ml_ops/maintenance_models
      training_data_path: data_nexus/maintenance_training_data.csv
      prediction_window_hours: 24
      failure_probability_threshold: 0.7
    alerting: # Dummy alerting config for services to initialize
      slack:
        enabled: false
      email:
        enabled: false
      rules: []
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_driver_realtime_data(self, driver_id: str) -> Dict[str, Any]:
        """
        Fetches a driver's real-time location, speed, current task, and vitals.
        """
        aggregated_telemetry = self.feature_store_client.get_feature(
            "aggregated_driver_telemetry", driver_id
        )
        driver_db_info = self.feature_store_client.get_feature(
            "driver", driver_id
        )  # Basic driver info (e.g., status, assigned order)

        data = {
            "driver_id": driver_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "latitude": (
                aggregated_telemetry.get("latitude", random.uniform(35.5, 35.9))
                if aggregated_telemetry
                else random.uniform(35.5, 35.9)
            ),
            "longitude": (
                aggregated_telemetry.get("longitude", random.uniform(51.2, 51.7))
                if aggregated_telemetry
                else random.uniform(51.2, 51.7)
            ),
            "speed_kph": (
                aggregated_telemetry.get("avg_speed_kph", random.uniform(0, 80))
                if aggregated_telemetry
                else random.uniform(0, 80)
            ),
            "status": (
                driver_db_info.get("status", "unknown") if driver_db_info else "unknown"
            ),
            "current_task": (
                driver_db_info.get("current_task", "idle") if driver_db_info else "idle"
            ),
            "assigned_order_id": (
                driver_db_info.get("assigned_order_id") if driver_db_info else None
            ),
            "fuel_level_percent": (
                aggregated_telemetry.get(
                    "avg_fuel_level_percent", random.uniform(10, 100)
                )
                if aggregated_telemetry
                else random.uniform(10, 100)
            ),
            "fatigue_score": self.wellbeing_monitor._calculate_fatigue_score(
                driver_id
            ),  # Directly call monitor's internal method
            "stress_score": self.wellbeing_monitor._calculate_stress_score(driver_id),
        }
        return data

    def _get_order_status_data(self, order_id: str) -> Dict[str, Any]:
        """
        Fetches real-time status and details for a specific order.
        """
        order_features = self.feature_store_client.get_feature(
            "order", order_id
        )  # Assumed feature group
        if not order_features:
            return {
                "order_id": order_id,
                "status": "NOT_FOUND",
                "details": "Order not found in feature store.",
            }

        # Mocking node coordinates from graph for origin/destination
        origin_node_id = order_features.get("origin_node")
        dest_node_id = order_features.get("destination_node")

        origin_coords = (
            (
                self.graph.nodes[origin_node_id]["y"],
                self.graph.nodes[origin_node_id]["x"],
            )
            if origin_node_id in self.graph.nodes()
            else (0, 0)
        )
        dest_coords = (
            (self.graph.nodes[dest_node_id]["y"], self.graph.nodes[dest_node_id]["x"])
            if dest_node_id in self.graph.nodes()
            else (0, 0)
        )

        return {
            "order_id": order_id,
            "status": order_features.get("status", "pending"),
            "origin_coords": origin_coords,
            "destination_coords": dest_coords,
            "pickup_window_start": order_features.get("pickup_time_start"),
            "delivery_window_latest": order_features.get("delivery_time_latest"),
            "current_driver_id": order_features.get("assigned_driver_id"),
        }

    def _get_fleet_overview_data(self) -> Dict[str, Any]:
        """
        Provides a high-level summary of the entire fleet.
        """
        all_drivers_telemetry_keys = self.feature_store_client.client.keys(
            "aggregated_driver_telemetry:*"
        )
        all_driver_ids = [k.split(":")[1] for k in all_drivers_telemetry_keys]

        active_drivers = 0
        idle_drivers = 0
        on_route_drivers = 0
        drivers_with_high_fatigue = 0
        drivers_needing_maintenance = 0

        for driver_id in all_driver_ids:
            driver_info = self.feature_store_client.get_feature("driver", driver_id)
            if driver_info:
                status = driver_info.get("status", "unknown")
                if status == "available":
                    idle_drivers += 1
                elif status == "on_route":
                    on_route_drivers += 1
                active_drivers += 1  # Any driver that exists is active for this count

            # Use internal methods of wellbeing/maintenance for data enrichment
            # (Note: In a real system, these would ideally be pre-calculated and stored in feature store)
            self.wellbeing_monitor._update_telemetry_history(
                driver_id,
                self.feature_store_client.get_feature(
                    "aggregated_driver_telemetry", driver_id
                )
                or {},
            )
            fatigue_score = self.wellbeing_monitor._calculate_fatigue_score(driver_id)
            if (
                fatigue_score
                > self.wellbeing_monitor.monitor_config["fatigue_score_threshold"]
            ):
                drivers_with_high_fatigue += 1

            # Simulate maintenance check
            # self.maintenance_service._update_sensor_history(driver_id, self.feature_store_client.get_feature("aggregated_driver_telemetry", driver_id) or {})
            # prediction_proba = self.maintenance_service.model.predict_proba(self.maintenance_service._prepare_features_for_prediction(driver_id))[0][1] if self.maintenance_service.model else 0
            # if prediction_proba > self.maintenance_service.maintenance_config['failure_probability_threshold']:
            #     drivers_needing_maintenance += 1

        all_orders = self.feature_store_client.stream_features_to_df(
            "order", pattern="*"
        )
        total_orders = len(all_orders)
        pending_orders = len(all_orders[all_orders["status"] == "pending"])
        delivered_orders = len(all_orders[all_orders["status"] == "delivered"])

        return {
            "total_drivers": active_drivers,
            "available_drivers": idle_drivers,
            "on_route_drivers": on_route_drivers,
            "drivers_high_fatigue": drivers_with_high_fatigue,
            "drivers_maintenance_needed": drivers_needing_maintenance,  # Placeholder for actual calculation
            "total_orders": total_orders,
            "pending_orders": pending_orders,
            "delivered_orders": delivered_orders,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

    async def get_ar_data_stream(
        self, requested_driver_id: str = None
    ) -> Dict[str, Any]:
        """
        Generates real-time data for an AR dashboard, focusing on a specific driver or fleet overview.
        """
        data = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "map_tile_server": self.ar_config["map_tile_server_url"],
            "overlays": [],
        }

        if requested_driver_id:
            # Focus on a single driver
            driver_data = self._get_driver_realtime_data(requested_driver_id)
            data["focus_driver"] = driver_data
            data["overlays"].append(
                {
                    "type": "driver_vehicle_3d_model",
                    "id": f"driver_asset_{requested_driver_id}",
                    "lat": driver_data["latitude"],
                    "lon": driver_data["longitude"],
                    "heading": random.uniform(0, 360),  # Mock heading
                    "speed_kph": driver_data["speed_kph"],
                    "color": (
                        "green" if driver_data["status"] == "available" else "blue"
                    ),
                    "label": f"Driver {requested_driver_id} ({driver_data['status']})",
                }
            )

            if driver_data["assigned_order_id"]:
                order_data = self._get_order_status_data(
                    driver_data["assigned_order_id"]
                )
                data["overlays"].append(
                    {
                        "type": "order_pickup_location",
                        "id": f"order_pickup_{order_data['order_id']}",
                        "lat": order_data["origin_coords"][0],
                        "lon": order_data["origin_coords"][1],
                        "label": f"Pickup {order_data['order_id']}",
                        "status": order_data["status"],
                    }
                )
                data["overlays"].append(
                    {
                        "type": "order_delivery_location",
                        "id": f"order_delivery_{order_data['order_id']}",
                        "lat": order_data["destination_coords"][0],
                        "lon": order_data["destination_coords"][1],
                        "label": f"Deliver {order_data['order_id']}",
                        "status": order_data["status"],
                    }
                )

            if driver_data["fatigue_score"] > 0.5:
                data["overlays"].append(
                    {
                        "type": "driver_health_warning",
                        "id": f"fatigue_alert_{requested_driver_id}",
                        "lat": driver_data["latitude"],
                        "lon": driver_data["longitude"],
                        "message": f"Fatigue: {driver_data['fatigue_score']:.1f}",
                        "severity": "warning",
                    }
                )

        else:
            # Fleet overview
            fleet_overview = self._get_fleet_overview_data()
            data["fleet_overview"] = fleet_overview

            # Show all active drivers on map
            all_driver_telemetry_keys = self.feature_store_client.client.keys(
                "aggregated_driver_telemetry:*"
            )
            all_driver_ids = [k.split(":")[1] for k in all_driver_telemetry_keys]

            for driver_id in all_driver_ids:
                driver_data = self._get_driver_realtime_data(driver_id)
                data["overlays"].append(
                    {
                        "type": "driver_location_marker",
                        "id": f"driver_marker_{driver_id}",
                        "lat": driver_data["latitude"],
                        "lon": driver_data["longitude"],
                        "label": f"Driver {driver_id}",
                        "status": driver_data["status"],
                        "color": (
                            "green"
                            if driver_data["status"] == "available"
                            else (
                                "blue"
                                if driver_data["status"] == "on_route"
                                else "gray"
                            )
                        ),
                    }
                )
                # Add summary alerts to the map, up to max_alert_overlays
                if (
                    driver_data["fatigue_score"] > 0.5
                    and len(data["overlays"]) < self.ar_config["max_alert_overlays"]
                ):
                    data["overlays"].append(
                        {
                            "type": "alert_icon",
                            "id": f"fatigue_alert_icon_{driver_id}",
                            "lat": driver_data["latitude"],
                            "lon": driver_data["longitude"],
                            "message": f"Fatigue ({driver_data['fatigue_score']:.1f})",
                            "icon": "exclamation-triangle",
                            "severity": "warning",
                        }
                    )
        return data

    async def run_dashboard_update_loop(self):
        if not self.ar_config["enabled"]:
            logger.info("AR Dashboard backend is disabled.")
            return

        update_interval = self.ar_config["update_interval_seconds"]
        logger.info("Starting AR Dashboard update loop...")

        # Manually create a dummy graph if osmnx_processor cannot create it in sandbox env
        graph_output_path = (
            "data_nexus/road_network_graph/preprocessed_tehran_graph.gml"
        )
        os.makedirs(os.path.dirname(graph_output_path), exist_ok=True)
        if not os.path.exists(graph_output_path):
            g_temp = nx.MultiDiGraph()
            for i in range(1, 5):
                g_temp.add_node(
                    i * 100, x=random.uniform(51.2, 51.7), y=random.uniform(35.5, 35.9)
                )
            g_temp.add_edge(
                100, 200, key=0, travel_time=60, length=500, traffic_factor=1.0
            )
            g_temp.add_edge(
                200, 300, key=0, travel_time=90, length=800, traffic_factor=1.0
            )
            g_temp.add_edge(
                300, 400, key=0, travel_time=40, length=300, traffic_factor=1.0
            )
            nx.write_gml(g_temp, graph_output_path)
            print("Dummy graph generated for AR Dashboard.")

        while True:
            try:
                # Simulate fetching data for a specific driver (e.g., driver_1) or fleet
                # For demo, we'll alternate between fleet view and a specific driver.
                if datetime.datetime.now().second % 10 < 5:
                    ar_data = await self.get_ar_data_stream(
                        requested_driver_id="driver_1"
                    )
                    print(f"\n--- AR Dashboard (Focus: Driver 1) ---")
                    if "focus_driver" in ar_data:
                        print(
                            f"Driver 1 (Lat: {ar_data['focus_driver']['latitude']:.4f}, Lon: {ar_data['focus_driver']['longitude']:.4f}), Speed: {ar_data['focus_driver']['speed_kph']:.1f} kph, Fatigue: {ar_data['focus_driver']['fatigue_score']:.1f}"
                        )
                    print(f"Overlays: {len(ar_data['overlays'])}")
                else:
                    ar_data = await self.get_ar_data_stream()
                    print(f"\n--- AR Dashboard (Fleet Overview) ---")
                    if "fleet_overview" in ar_data:
                        print(
                            f"Total Drivers: {ar_data['fleet_overview']['total_drivers']}, Fatigue Alerts: {ar_data['fleet_overview']['drivers_high_fatigue']}"
                        )
                    print(f"Overlays: {len(ar_data['overlays'])}")

                # In a real system, this `ar_data` would be sent via WebSocket to a frontend.
                # json_output = json.dumps(ar_data, indent=2, ensure_ascii=False)
                # print(json_output) # For debugging

            except Exception as e:
                logger.error(f"Error in AR Dashboard update loop: {e}", exc_info=True)

            await asyncio.sleep(update_interval)


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    osm_config_path_for_ar = (
        "conf/environments/dev.yaml"  # Uses this path to look up OSM config
    )
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
    ar_dashboard:
      enabled: true
      update_interval_seconds: 2
      map_tile_server_url: https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
      realtime_overlay_refresh_rate_ms: 500
      max_alert_overlays: 3
    osm_processing_config_path: conf/osm_processing_config.yaml
    driver_wellbeing_monitor:
      enabled: true
      history_window_minutes: 60
      fatigue_score_threshold: 0.5
      stress_score_threshold: 0.6
      speeding_duration_threshold_minutes: 5
      harsh_braking_count_threshold: 3
    predictive_maintenance:
      enabled: true
      history_window_minutes: 240
      model_dir: ml_ops/maintenance_models
      training_data_path: data_nexus/maintenance_training_data.csv
      prediction_window_hours: 24
      failure_probability_threshold: 0.7
    alerting:
      slack:
        enabled: false
        webhook_url: http://mock-slack-url
      email:
        enabled: false
        sender_email: mock@example.com
        sender_password: mock
        recipients: [mock_recipient@example.com]
      rules:
        - name: DriverFatigueAlert
          metric: driver_fatigue_score
          threshold: 0.5
          operator: ">"
          window_minutes: 1
          severity: warning
          cooldown_minutes: 5
          channels: ["slack"]
        - name: DriverStressAlert
          metric: driver_stress_score
          threshold: 0.6
          operator: ">"
          window_minutes: 1
          severity: critical
          cooldown_minutes: 5
          channels: ["slack", "email"]
"""
            )
    # Dummy OSM config if not exists
    if not os.path.exists("conf/osm_processing_config.yaml"):
        with open("conf/osm_processing_config.yaml", "w") as f:
            f.write(
                """
osm_data:
  pbf_path: data_nexus/raw_osm_data/tehran_iran.osm.pbf
  bounding_box: [35.5, 51.2, 35.9, 51.7]
  cache_dir: data_nexus/road_network_graph/
graph_serialization:
  output_format: gml
  output_path: data_nexus/road_network_graph/preprocessed_tehran_graph.gml
osmnx:
  graph_type: drive
  network_type: drive_service
preprocessing_steps: []
"""
            )

    # Populate Redis with dummy aggregated telemetry for testing
    import redis

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0)
        r.ping()
        print("Connected to Redis. Populating dummy telemetry for AR Dashboard.")

        # Driver 1: High fatigue (for alert demo)
        for i in range(12):
            ts = (
                datetime.datetime.utcnow()
                - datetime.timedelta(hours=6)
                + datetime.timedelta(minutes=i * 30)
            )
            r.set(
                f"aggregated_driver_telemetry:driver_1",
                json.dumps(
                    {
                        "timestamp_utc": ts.isoformat() + "Z",
                        "latitude": 35.7 + i * 0.001,
                        "longitude": 51.4 + i * 0.001,
                        "last_seen_node_id": 100 + i,
                        "avg_speed_kph": 70.0 + random.uniform(-5, 5),
                        "fuel_level_percent": 90.0 - i * 5,
                        "avg_load_kg": 10.0,
                    }
                ),
            )
        r.set(
            "driver:driver_1",
            json.dumps(
                {
                    "driver_id": "driver_1",
                    "status": "on_route",
                    "current_task": "delivering",
                    "assigned_order_id": "ORD_D1_001",
                }
            ),
        )
        r.set(
            "order:ORD_D1_001",
            json.dumps(
                {
                    "order_id": "ORD_D1_001",
                    "status": "picked_up",
                    "origin_node": 100,
                    "destination_node": 400,
                }
            ),
        )

        # Driver 2: Normal
        ts = datetime.datetime.utcnow()
        r.set(
            f"aggregated_driver_telemetry:driver_2",
            json.dumps(
                {
                    "timestamp_utc": ts.isoformat() + "Z",
                    "latitude": 35.75,
                    "longitude": 51.45,
                    "last_seen_node_id": 200,
                    "avg_speed_kph": 40.0,
                    "fuel_level_percent": 60.0,
                    "avg_load_kg": 5.0,
                }
            ),
        )
        r.set(
            "driver:driver_2",
            json.dumps(
                {"driver_id": "driver_2", "status": "available", "current_task": "idle"}
            ),
        )

    except redis.exceptions.ConnectionError:
        print("Redis not running. AR Dashboard will start with empty data.")

    async def main_ar():
        dashboard = AugmentedRealityDashboard(config_file)
        print("Starting AR Dashboard backend for 30 seconds...")
        try:
            await asyncio.wait_for(dashboard.run_dashboard_update_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("\nAR Dashboard backend timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nAR Dashboard backend stopped by user.")

    import asyncio

    asyncio.run(main_ar())
