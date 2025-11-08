import datetime
import json
import yaml
import os
import logging
import random
import networkx as nx
import numpy as np

from feature_forge.feature_store_client import (
    FeatureStoreClient,
)  # To get real-time features
from routing.astar_optimization_logic import AStarRouting  # To estimate route costs
from routing.osmnx_processor import OSMNxProcessor  # For graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicPricingAndIncentiveEngine:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.pricing_config = self.config["environments"][environment][
            "economic_incentives"
        ]

        self.feature_store_client = FeatureStoreClient(config_path, environment)
        self.osm_processor = OSMNxProcessor(
            self.config["environments"][environment].get(
                "osm_processing_config_path", "conf/osm_processing_config.yaml"
            )
        )
        self.graph = self.osm_processor.get_graph()
        self.router = AStarRouting(
            self.graph,
            self.config["environments"][environment].get(
                "routing_config_path", "conf/routing_engine_config.yaml"
            ),
        )

        logger.info("DynamicPricingAndIncentiveEngine initialized.")

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
    economic_incentives:
      base_price_per_km: 0.5
      base_incentive_per_km: 0.2
      demand_surge_factor: 1.5
      driver_shortage_factor: 2.0
      urgency_multiplier: 1.2
      min_price: 2.0
      max_incentive: 10.0
      demand_threshold_high: 0.8
      driver_availability_threshold_low: 0.3
      surge_window_minutes: 10
      lookahead_minutes_for_demand: 15
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_realtime_context(self, origin_node_id: int, destination_node_id: int):
        # Fetch dynamic features from Redis
        node_demand_features = self.feature_store_client.get_feature(
            "node_demand", str(origin_node_id)
        )
        driver_dynamic_features = self.feature_store_client.stream_features_to_df(
            "driver_dynamic", pattern="*"
        )  # Get all drivers for availability

        current_time = datetime.datetime.now(datetime.timezone.utc)

        # Calculate traffic factor along the route
        route_info = self.router.get_shortest_path(origin_node_id, destination_node_id)
        estimated_distance_km = route_info["distance"] / 1000 if route_info else 0.0
        estimated_travel_time_min = (
            route_info["travel_time"] / 60 if route_info else 0.0
        )

        # Aggregate current traffic factors on the route
        avg_traffic_factor = 1.0  # Default
        if route_info and route_info["path"]:
            traffic_factors = []
            for i in range(len(route_info["path"]) - 1):
                u, v = route_info["path"][i], route_info["path"][i + 1]
                edge_data = self.graph[u][v][0]  # Assuming key=0 for simplicity
                traffic_factors.append(edge_data.get("traffic_factor", 1.0))
            if traffic_factors:
                avg_traffic_factor = np.mean(traffic_factors)

        # Demand surge near origin
        num_new_orders_5min = (
            node_demand_features.get("num_new_orders_5min", 0)
            if node_demand_features
            else 0
        )
        is_demand_surge = (
            num_new_orders_5min / self.pricing_config["surge_window_minutes"]
            > self.pricing_config["demand_threshold_high"]
        )

        # Driver availability
        available_drivers_count = (
            sum(
                1
                for _, row in driver_dynamic_features.iterrows()
                if row.get("status") == "available"
            )
            if not driver_dynamic_features.empty
            else 0
        )
        total_drivers_count = (
            len(driver_dynamic_features) if not driver_dynamic_features.empty else 1
        )
        driver_availability_ratio = available_drivers_count / total_drivers_count
        is_driver_shortage = (
            driver_availability_ratio
            < self.pricing_config["driver_availability_threshold_low"]
        )

        return {
            "estimated_distance_km": estimated_distance_km,
            "estimated_travel_time_min": estimated_travel_time_min,
            "avg_traffic_factor": avg_traffic_factor,
            "is_demand_surge": is_demand_surge,
            "is_driver_shortage": is_driver_shortage,
            "current_time": current_time,
        }

    def calculate_dynamic_price(
        self,
        order_id: str,
        origin_node_id: int,
        destination_node_id: int,
        requested_pickup_time: datetime.datetime,
    ) -> float:
        """
        Calculates the dynamic price for a customer order.
        """
        context = self._get_realtime_context(origin_node_id, destination_node_id)

        base_price = (
            context["estimated_distance_km"] * self.pricing_config["base_price_per_km"]
        )

        # Adjust for demand surge
        if context["is_demand_surge"]:
            base_price *= self.pricing_config["demand_surge_factor"]
            logger.debug(f"Price for {order_id} increased due to demand surge.")

        # Adjust for urgency
        time_to_pickup_min = (
            requested_pickup_time - context["current_time"]
        ).total_seconds() / 60
        if (
            time_to_pickup_min < self.pricing_config["lookahead_minutes_for_demand"]
        ):  # Urgent if within lookahead
            base_price *= self.pricing_config["urgency_multiplier"]
            logger.debug(f"Price for {order_id} increased due to urgency.")

        # Adjust for traffic (more traffic, higher price)
        base_price *= context["avg_traffic_factor"]

        final_price = max(base_price, self.pricing_config["min_price"])
        logger.info(
            f"Calculated dynamic price for order {order_id}: ${final_price:.2f}"
        )
        return final_price

    def calculate_dynamic_incentive(
        self,
        driver_id: str,
        order_id: str,
        origin_node_id: int,
        destination_node_id: int,
        actual_payout_customer: float,
    ) -> float:
        """
        Calculates an additional incentive for a driver based on real-time factors.
        """
        context = self._get_realtime_context(origin_node_id, destination_node_id)

        base_incentive = (
            context["estimated_distance_km"]
            * self.pricing_config["base_incentive_per_km"]
        )

        # Adjust for driver shortage
        if context["is_driver_shortage"]:
            base_incentive *= self.pricing_config["driver_shortage_factor"]
            logger.debug(f"Incentive for {driver_id} increased due to driver shortage.")

        # Adjust for difficult traffic conditions (more traffic, higher incentive)
        # This could be inverse to how price is adjusted, or directly proportional for driver
        base_incentive *= context["avg_traffic_factor"]

        # If the customer paid a very high price, a fraction could be passed as incentive
        incentive_from_customer_payout = (
            actual_payout_customer
            - (
                context["estimated_distance_km"]
                * self.pricing_config["base_price_per_km"]
            )
        ) * 0.2  # 20% of surge
        if incentive_from_customer_payout > 0:
            base_incentive += incentive_from_customer_payout
            logger.debug(
                f"Incentive for {driver_id} increased by {incentive_from_customer_payout:.2f} from customer surge."
            )

        final_incentive = min(base_incentive, self.pricing_config["max_incentive"])
        logger.info(
            f"Calculated dynamic incentive for driver {driver_id} (order {order_id}): ${final_incentive:.2f}"
        )
        return final_incentive


if __name__ == "__main__":
    # Ensure dummy config files and dependencies exist
    config_file = "conf/environments/dev.yaml"
    osm_config_path = "conf/osm_processing_config.yaml"
    routing_config_path = "conf/routing_engine_config.yaml"

    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(osm_config_path), exist_ok=True)
    os.makedirs(os.path.dirname(routing_config_path), exist_ok=True)

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
    feature_store:
      prefix_driver_dynamic: driver_dynamic_features:
      prefix_node_demand: node_demand_features:
    economic_incentives:
      base_price_per_km: 0.5
      base_incentive_per_km: 0.2
      demand_surge_factor: 1.5
      driver_shortage_factor: 2.0
      urgency_multiplier: 1.2
      min_price: 2.0
      max_incentive: 10.0
      demand_threshold_high: 0.01 # Lower for easier trigger in demo
      driver_availability_threshold_low: 0.8 # Higher for easier trigger in demo
      surge_window_minutes: 10
      lookahead_minutes_for_demand: 15
"""
            )
    # Dummy osm config for graph generation if not existing
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
    # Dummy routing config
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

    # Manually create a dummy graph if osmnx_processor cannot create it in sandbox env
    graph_output_path = "data_nexus/road_network_graph/preprocessed_tehran_graph.gml"
    os.makedirs(os.path.dirname(graph_output_path), exist_ok=True)
    if not os.path.exists(graph_output_path):
        g_temp = nx.MultiDiGraph()
        for i in range(1, 5):
            g_temp.add_node(
                i * 100, x=random.uniform(51.2, 51.7), y=random.uniform(35.5, 35.9)
            )
        g_temp.add_edge(100, 200, key=0, travel_time=60, length=500, traffic_factor=1.0)
        g_temp.add_edge(200, 300, key=0, travel_time=90, length=800, traffic_factor=1.0)
        g_temp.add_edge(300, 400, key=0, travel_time=40, length=300, traffic_factor=1.0)
        nx.write_gml(g_temp, graph_output_path)
        print("Dummy graph generated for Dynamic Pricing.")

    # Populate Redis with dummy feature data for testing
    import redis

    r = redis.StrictRedis(host="localhost", port=6379, db=0)
    try:
        r.ping()
        print("Connected to Redis. Populating dummy features.")
        r.set(
            "node_demand_features:100",
            json.dumps(
                {
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                    "num_new_orders_5min": 10,  # High demand
                    "total_weight_5min": 50,
                    "total_volume_5min": 5,
                }
            ),
        )
        r.set(
            "driver_dynamic_features:driver_A",
            json.dumps(
                {
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                    "avg_speed_5min": 40,
                    "avg_load_5min": 10,
                    "status": "on_route",
                }
            ),
        )
        r.set(
            "driver_dynamic_features:driver_B",
            json.dumps(
                {
                    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "avg_speed_5min": 30,
                    "avg_load_5min": 5,
                    "status": "available",
                }
            ),
        )
        r.set(
            "traffic:edge:100-200-0",
            json.dumps(
                {
                    "current_travel_time": 90,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                }
            ),
        )
        # Ensure graph has 'traffic_factor' field after router updates it.
        # For this demo, manually inject a traffic factor in the graph directly.
        temp_graph = nx.read_gml(graph_output_path)
        temp_graph[100][200][0]["traffic_factor"] = 1.5  # Simulate congestion
        temp_graph[200][300][0]["traffic_factor"] = 1.2  # Some traffic
        nx.write_gml(temp_graph, graph_output_path)
    except redis.exceptions.ConnectionError:
        print("Redis not running. Dynamic pricing will start with empty feature data.")

    engine = DynamicPricingAndIncentiveEngine(config_file)

    # --- Test 1: Calculate dynamic price for an order ---
    order_id = "ORD_X_123"
    origin = 100
    destination = 300
    pickup_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=5
    )  # Urgent pickup

    print(f"\n--- Calculating Dynamic Price for Order {order_id} ---")
    dynamic_price = engine.calculate_dynamic_price(
        order_id, origin, destination, pickup_time
    )
    print(f"Customer would pay: ${dynamic_price:.2f}")

    # --- Test 2: Calculate dynamic incentive for a driver ---
    driver_id = "driver_B"
    # Assume customer paid the calculated dynamic price
    customer_paid_price = dynamic_price

    print(
        f"\n--- Calculating Dynamic Incentive for Driver {driver_id} for Order {order_id} ---"
    )
    dynamic_incentive = engine.calculate_dynamic_incentive(
        driver_id, order_id, origin, destination, customer_paid_price
    )
    print(
        f"Driver {driver_id} would receive an additional incentive of: ${dynamic_incentive:.2f}"
    )

    print("\nDynamic Pricing and Incentive Engine demo complete.")
