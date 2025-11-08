import asyncio
import datetime
import yaml
import os
import logging
import json
import random
import networkx as nx

from feature_forge.feature_store_client import FeatureStoreClient
from routing.osmnx_processor import OSMNxProcessor
from data_nexus.simulation_scenarios.simpy_delivery_environment import (
    SimpyDeliveryEnvironment,
    Driver,
    Order,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DigitalTwinSynchronizer:
    def __init__(
        self,
        config_path="conf/environments/prod.yaml",
        environment="dev",
        sim_config_path="conf/environments/dev.yaml",
    ):
        self.config = self._load_config(config_path)
        self.env_config = self.config["environments"][environment]
        self.sim_config_path = sim_config_path

        self.feature_store_client = FeatureStoreClient(config_path, environment)
        self.osm_processor = OSMNxProcessor(
            self.env_config.get(
                "osm_processing_config_path", "conf/osm_processing_config.yaml"
            )
        )
        self.current_graph = self.osm_processor.get_graph()

        # Initialize a SimpyDeliveryEnvironment as the digital twin
        # This will be reset and populated with real-time data
        try:
            self.digital_twin_env = SimpyDeliveryEnvironment(
                sim_config_path,
                scenario_path=None,  # We'll populate drivers and orders manually
                router=None,  # Will be set dynamically based on current graph/traffic
                rl_agent_inferer=None,  # Digital twin might use current policy or simulate behavior
            )
        except Exception as e:
            logger.warning(
                f"Could not initialize SimpyDeliveryEnvironment: {e}. Proceeding without digital twin."
            )
            self.digital_twin_env = None
        logger.info("DigitalTwinSynchronizer initialized.")

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
    osm_processing_config_path: conf/osm_processing_config.yaml
    routing_config_path: conf/routing_engine_config.yaml
    digital_twin:
      update_interval_seconds: 10
      lookahead_minutes: 60
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _create_mock_router(self, graph, routing_config_path):
        from routing.astar_optimization_logic import AStarRouting

        return AStarRouting(graph, routing_config_path)

    async def synchronize_digital_twin(self):
        """
        Fetches current real-world state and updates the digital twin simulation environment.
        """
        if self.digital_twin_env is None:
            logger.warning(
                "Digital twin environment not initialized. Skipping synchronization."
            )
            return
        logger.info("Synchronizing digital twin with real-time data...")

        # 1. Update Graph and Router with latest traffic from Feature Store
        traffic_keys = self.feature_store_client.client.keys("traffic:edge:*")
        updated_graph = self.current_graph.copy()
        if traffic_keys:
            traffic_data = self.feature_store_client.client.mget(traffic_keys)
            for key_str, traffic_json in zip(traffic_keys, traffic_data):
                if traffic_json:
                    traffic_info = json.loads(traffic_json)
                    edge_parts = key_str.split(":")[1].split("-")
                    u, v, k = int(edge_parts[0]), int(edge_parts[1]), int(edge_parts[2])

                    if updated_graph.has_edge(u, v, key=k):
                        updated_graph[u][v][k]["current_predicted_travel_time"] = (
                            traffic_info["current_travel_time"]
                        )
        self.current_graph = updated_graph
        self.digital_twin_env.router = self._create_mock_router(
            self.current_graph, self.env_config["routing_config_path"]
        )

        # 2. Fetch all drivers from Feature Store / DB
        driver_telemetry_keys = self.feature_store_client.client.keys(
            "aggregated_driver_telemetry:*"
        )
        driver_ids = [
            k.split("aggregated_driver_telemetry:")[1] for k in driver_telemetry_keys
        ]
        real_drivers_data = {}
        if driver_ids:
            raw_telemetry_data = self.feature_store_client.client.mget(
                driver_telemetry_keys
            )
            for i, data in enumerate(raw_telemetry_data):
                if data:
                    real_drivers_data[driver_ids[i]] = json.loads(data)

        dt_drivers = []
        for driver_id, drv_data in real_drivers_data.items():
            current_node = drv_data.get(
                "last_seen_node_id", random.choice(list(self.current_graph.nodes()))
            )

            # Map aggregated telemetry to a SimpyDeliveryEnvironment.Driver object
            dt_driver = Driver(
                driver_id=driver_id,
                start_node=current_node,
                vehicle_type="car",  # Assuming for now
                capacity=100.0,  # Placeholder
                speed_mps=drv_data.get("avg_speed_kph", 40) / 3.6,  # Convert kph to mps
            )
            # Additional states like current_load, assigned_orders should be populated from DB/Feature Store
            dt_drivers.append(dt_driver)
        self.digital_twin_env.drivers = {d.driver_id: d for d in dt_drivers}

        # 3. Fetch active orders from DB / Feature Store
        # For this demo, let's assume we fetch some existing orders
        # In a real system, you'd fetch from DatabaseManager and potentially also upcoming orders from Kafka

        dt_orders = []
        # Mock 5 new pending orders for the digital twin for lookahead analysis
        num_new_orders_for_dt = 5
        nodes = list(self.current_graph.nodes())
        for i in range(num_new_orders_for_dt):
            origin_node = random.choice(nodes)
            dest_node = random.choice(nodes)
            while dest_node == origin_node:
                dest_node = random.choice(nodes)

            # Use current time + some lookahead for new orders in digital twin
            future_pickup = datetime.datetime.utcnow() + datetime.timedelta(
                minutes=random.randint(
                    5, self.env_config["digital_twin"]["lookahead_minutes"]
                )
            )

            dt_order = Order(
                order_id=f"DT_ORDER_{int(future_pickup.timestamp())}_{i}",
                origin_node=origin_node,
                destination_node=dest_node,
                pickup_time_start=future_pickup,
                pickup_time_end=future_pickup + datetime.timedelta(minutes=15),
                delivery_time_latest=future_pickup + datetime.timedelta(hours=2),
                weight=random.uniform(1, 10),
                volume=random.uniform(0.1, 1.0),
            )
            dt_orders.append(dt_order)
        self.digital_twin_env.orders = {o.order_id: o for o in dt_orders}

        # Reset Simpy environment with current data
        self.digital_twin_env.reset_simpy_env(
            initial_drivers=list(self.digital_twin_env.drivers.values()),
            initial_orders=list(self.digital_twin_env.orders.values()),
        )
        logger.info(
            f"Digital Twin synchronized. Drivers: {len(dt_drivers)}, Orders: {len(dt_orders)} (including lookahead)."
        )

    async def run_digital_twin_loop(self):
        """
        Periodically synchronizes the digital twin and runs a short simulation
        for lookahead analysis or shadow mode.
        """
        update_interval = self.env_config.get("digital_twin", {}).get("update_interval_seconds", 10)
        lookahead_minutes = self.env_config.get("digital_twin", {}).get("lookahead_minutes", 60)

        while True:
            try:
                await self.synchronize_digital_twin()

                # Run the digital twin for lookahead_minutes
                logger.info(
                    f"Running digital twin simulation for {lookahead_minutes} minutes..."
                )
                metrics_df = self.digital_twin_env.run_simulation(
                    until=lookahead_minutes * 60
                )

                if not metrics_df.empty:
                    final_metrics = metrics_df.iloc[-1]
                    logger.info(
                        f"Digital Twin Lookahead Results at {lookahead_minutes} min:"
                    )
                    logger.info(
                        f"  Delivered Orders: {final_metrics.get('num_delivered_orders', 0)}"
                    )
                    logger.info(
                        f"  Pending Orders: {final_metrics.get('num_pending_orders', 0)}"
                    )
                    logger.info(
                        f"  Total Driver Distance: {final_metrics.get('total_driver_distance', 0):.2f}m"
                    )
                else:
                    logger.warning("Digital Twin simulation produced no metrics.")

                # This is where shadow testing results would be compared to real-world,
                # or where alternative policy scenarios would be evaluated.

            except Exception as e:
                logger.critical(f"Error in digital twin loop: {e}", exc_info=True)

            await asyncio.sleep(update_interval)


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    osm_config_path = "conf/osm_processing_config.yaml"
    routing_config_path = "conf/routing_engine_config.yaml"

    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(osm_config_path), exist_ok=True)
    os.makedirs(os.path.dirname(routing_config_path), exist_ok=True)

    # Create dummy environment config
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
    osm_processing_config_path: conf/osm_processing_config.yaml
    routing_config_path: conf/routing_engine_config.yaml
    digital_twin:
      update_interval_seconds: 10
      lookahead_minutes: 5 # Short lookahead for demo
"""
            )

    # Create dummy OSM processing config
    if not os.path.exists(osm_config_path):
        with open(osm_config_path, "w") as f:
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

    # Create dummy routing config
    if not os.path.exists(routing_config_path):
        with open(routing_config_path, "w") as f:
            f.write(
                """
graphhopper:
  host: http://graphhopper-service:8989/
  api_key: dummy_key
  profile: car
astar_custom_params:
  weight_factor_traffic: 1.5
  weight_factor_distance: 1.0
"""
            )

    # This requires Redis to be running and `kafka_producer_simulator.py` to push telemetry
    # and `src/data_nexus/realtime_telemetry_aggregator.py` to aggregate it into Redis
    # before `DigitalTwinSynchronizer` can consume it.

    # Simulate populating Redis with some aggregated telemetry
    import redis

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0)
        r.ping()
        print("Connected to Redis. Populating dummy telemetry for Digital Twin.")
        r.set(
            "aggregated_driver_telemetry:driver_1",
            json.dumps(
                {
                    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "last_seen_node_id": 100,
                    "avg_speed_kph": 50.0,
                    "fuel_level_percent": 75.0,
                }
            ),
        )
        r.set(
            "aggregated_driver_telemetry:driver_2",
            json.dumps(
                {
                    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "last_seen_node_id": 200,
                    "avg_speed_kph": 30.0,
                    "fuel_level_percent": 40.0,
                }
            ),
        )
        r.set(
            "traffic:edge:100-101-0",
            json.dumps(
                {
                    "current_travel_time": 120,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
    except redis.exceptions.ConnectionError:
        print("Redis not running. Digital Twin will start with empty data.")

    async def main():
        # Ensure dummy SimpyDeliveryEnvironment dependencies are available for execution
        # (This requires `src/fleet_simulator/simpy_delivery_environment.py` etc. to be available)
        # For simplicity, if these are not available, this example will throw.
        # In a real sandbox, these might be mocked or pre-provided.
        # This execution will assume SimpyDeliveryEnvironment can be imported.

        logger.info("Checking if dummy graph needs to be created for Digital Twin.")
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
            g_temp.add_edge(100, 200, key=0, travel_time=60, length=500)
            g_temp.add_edge(200, 300, key=0, travel_time=90, length=800)
            g_temp.add_edge(300, 400, key=0, travel_time=40, length=300)
            nx.write_gml(g_temp, graph_output_path)
            print("Dummy graph generated for Digital Twin.")

        dt_sync = DigitalTwinSynchronizer(config_file)
        print("Starting Digital Twin Synchronizer for 30 seconds...")
        try:
            await asyncio.wait_for(dt_sync.run_digital_twin_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("Digital Twin Synchronizer timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("Digital Twin Synchronizer stopped by user.")

    import asyncio

    asyncio.run(main())
