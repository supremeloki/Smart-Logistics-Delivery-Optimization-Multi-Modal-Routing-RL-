import datetime
import json
import yaml
import os
import logging
import random
import asyncio
from typing import Dict, Any, List
import redis
import networkx as nx


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "product:PROD_A",
            json.dumps(
                {
                    "current_stock": 100,
                    "production_rate": 10,
                    "lead_time_days": 5,
                    "criticality": "high",
                }
            ),
        )
        self.client.set(
            "warehouse:WH_001",
            json.dumps(
                {
                    "location": {"lat": 35.7, "lon": 51.4},
                    "capacity": 1000,
                    "current_occupancy": 500,
                }
            ),
        )
        self.client.set(
            "supplier:SUP_001",
            json.dumps(
                {
                    "location": {"lat": 34.0, "lon": 50.0},
                    "product_types": ["PROD_A"],
                    "reliability": 0.9,
                }
            ),
        )
        self.client.set(
            "route:WH_001-SUP_001",
            json.dumps({"travel_time_hours": 24, "capacity": 50}),
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


class MockRiskPredictor:  # Predicts likelihood and severity of disruptions
    def __init__(self):
        pass

    def predict_disruption_risk(
        self, location_data: Dict[str, Any], entity_type: str
    ) -> Dict[str, Any]:
        risk_score = random.uniform(0.01, 0.1)  # Base risk
        if random.random() < 0.1:  # 10% chance of higher risk
            risk_score = random.uniform(0.3, 0.7)
        return {
            "likelihood": risk_score,
            "severity_impact_percent": risk_score * 50,
        }  # e.g., 50% impact


class MockAlertManager:
    def __init__(self, config_path, environment):
        pass

    def _send_notifications(self, alert_details: Dict[str, Any], channels: List[str]):
        logging.warning(
            f"Mock Alert Sent: {alert_details['description']} (Severity: {alert_details['severity']}) via {channels}"
        )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisruptionSimulator:
    def __init__(
        self,
        config_path="conf/environments/prod.yaml",
        environment="dev",
        sc_graph_path="data_nexus/supply_chain_graph.gml",
    ):
        self.config = self._load_config(config_path)
        self.sim_config = self.config["environments"][environment][
            "disruption_simulator"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.risk_predictor = MockRiskPredictor()
        self.alert_manager = MockAlertManager(config_path, environment)

        self.supply_chain_graph = self._load_or_generate_sc_graph(sc_graph_path)
        self.active_disruptions = {}  # entity_id -> disruption_details

        logger.info("DisruptionSimulator initialized.")

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
    disruption_simulator:
      enabled: true
      simulation_interval_seconds: 10
      disruption_probability_base: 0.05 # Per entity per interval
      disruption_types: ["factory_fire", "port_closure", "major_traffic_jam", "supplier_delay"]
      impact_factors:
        factory_fire: {"stock_reduction_percent": 0.5, "lead_time_increase_days": 10}
        port_closure: {"route_capacity_reduction_percent": 0.8, "travel_time_increase_hours": 48}
        major_traffic_jam: {"route_travel_time_multiplier": 2.0}
        supplier_delay: {"lead_time_increase_days": 7, "production_rate_reduction_percent": 0.2}
    alerting:
      slack:
        enabled: false
      email:
        enabled: false
      rules: []
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_or_generate_sc_graph(self, graph_path):
        if os.path.exists(graph_path):
            return nx.read_gml(graph_path)
        else:
            logger.warning(
                f"SC Graph not found at {graph_path}. Generating a small dummy graph."
            )
            G = nx.DiGraph()
            G.add_nodes_from(
                [
                    (
                        "WH_001",
                        {"type": "warehouse", "location": {"lat": 35.7, "lon": 51.4}},
                    ),
                    (
                        "SUP_001",
                        {"type": "supplier", "location": {"lat": 34.0, "lon": 50.0}},
                    ),
                    (
                        "PORT_001",
                        {"type": "port", "location": {"lat": 36.0, "lon": 51.5}},
                    ),
                    (
                        "MFG_001",
                        {
                            "type": "manufacturer",
                            "location": {"lat": 34.5, "lon": 50.5},
                        },
                    ),
                ]
            )
            G.add_edges_from(
                [
                    (
                        "SUP_001",
                        "MFG_001",
                        {
                            "product": "raw_material_A",
                            "travel_time_hours": 12,
                            "capacity_units_day": 100,
                        },
                    ),
                    (
                        "MFG_001",
                        "WH_001",
                        {
                            "product": "PROD_A",
                            "travel_time_hours": 24,
                            "capacity_units_day": 50,
                        },
                    ),
                    (
                        "PORT_001",
                        "WH_001",
                        {
                            "product": "PROD_B",
                            "travel_time_hours": 6,
                            "capacity_units_day": 200,
                        },
                    ),
                ]
            )
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            nx.write_gml(G, graph_path)
            return G

    def _apply_disruption_impact(
        self, entity_id: str, entity_type: str, disruption_type: str, severity: float
    ):
        """Applies the simulated disruption impact to relevant entities in feature store."""
        impact_config = self.sim_config["impact_factors"].get(disruption_type, {})

        if entity_type == "supplier" and disruption_type == "factory_fire":
            supplier_data = self.feature_store.get_feature("supplier", entity_id)
            if supplier_data:
                # Reduce production capacity/stock directly from supplier
                initial_reliability = supplier_data.get("reliability", 1.0)
                supplier_data["reliability"] = max(
                    0.1,
                    initial_reliability
                    - severity * impact_config.get("stock_reduction_percent", 0.0),
                )
                self.feature_store.set_feature("supplier", entity_id, supplier_data)
                # Also impact related products
                for prod_id in supplier_data.get("product_types", []):
                    prod_data = self.feature_store.get_feature("product", prod_id)
                    if prod_data:
                        prod_data["current_stock"] = max(
                            0,
                            prod_data["current_stock"]
                            * (
                                1
                                - severity
                                * impact_config.get("stock_reduction_percent", 0.0)
                            ),
                        )
                        prod_data["lead_time_days"] += severity * impact_config.get(
                            "lead_time_increase_days", 0
                        )
                        self.feature_store.set_feature("product", prod_id, prod_data)
                logger.warning(
                    f"Disruption: {disruption_type} at {entity_id}. Impact: stock reduced, lead time increased."
                )

        elif entity_type == "port" and disruption_type == "port_closure":
            # Impact all routes through this port
            for u, v, data in self.supply_chain_graph.edges(data=True):
                if u == entity_id or v == entity_id:  # Any route touching this port
                    route_key = f"{u}-{v}"
                    route_data = (
                        self.feature_store.get_feature("route", route_key) or data
                    )  # Use graph data as fallback
                    if route_data:
                        route_data["capacity"] = max(
                            0,
                            route_data.get("capacity", 0)
                            * (
                                1
                                - severity
                                * impact_config.get(
                                    "route_capacity_reduction_percent", 0.0
                                )
                            ),
                        )
                        route_data["travel_time_hours"] += severity * impact_config.get(
                            "travel_time_increase_hours", 0
                        )
                        self.feature_store.set_feature("route", route_key, route_data)
            logger.warning(
                f"Disruption: {disruption_type} at {entity_id}. Impact: routes through port capacity/time affected."
            )

        elif entity_type == "route" and disruption_type == "major_traffic_jam":
            route_data = self.feature_store.get_feature("route", entity_id)
            if route_data:
                route_data["travel_time_hours"] *= impact_config.get(
                    "route_travel_time_multiplier", 1.0
                ) * (
                    1 + severity * 0.5
                )  # Scale multiplier by severity
                self.feature_store.set_feature("route", entity_id, route_data)
            logger.warning(
                f"Disruption: {disruption_type} on {entity_id}. Impact: route travel time heavily increased."
            )

        # Store active disruption
        self.active_disruptions[entity_id] = {
            "disruption_type": disruption_type,
            "severity": severity,
            "start_time": datetime.datetime.utcnow(),
            "estimated_end_time": datetime.datetime.utcnow()
            + datetime.timedelta(hours=random.randint(12, 72)),  # Random duration
        }

        self.alert_manager._send_notifications(
            {
                "rule_name": f"SupplyChainDisruption_{entity_id}",
                "entity_id": entity_id,
                "description": f"Detected {disruption_type} at {entity_id} with severity {severity:.2f}",
                "severity": "critical",
            },
            ["slack", "email"],
        )

    async def simulate_disruptions_loop(self):
        """Periodically simulates and applies supply chain disruptions."""
        if not self.sim_config["enabled"]:
            logger.info("Disruption simulator is disabled.")
            return

        while True:
            logger.info("Simulating potential supply chain disruptions...")

            # Check all key entities in the graph (suppliers, warehouses, ports, manufacturers, routes)
            for node_id, node_data in self.supply_chain_graph.nodes(data=True):
                entity_type = node_data.get("type")
                if entity_type:
                    risk_prediction = self.risk_predictor.predict_disruption_risk(
                        node_data.get("location", {}), entity_type
                    )

                    if (
                        random.random()
                        < self.sim_config["disruption_probability_base"]
                        + risk_prediction["likelihood"]
                    ):
                        disruption_type = random.choice(
                            self.sim_config["disruption_types"]
                        )
                        severity = (
                            risk_prediction["severity_impact_percent"] / 100.0
                        )  # Convert to 0-1 scale

                        if (
                            node_id not in self.active_disruptions
                        ):  # Only apply new disruptions
                            self._apply_disruption_impact(
                                node_id, entity_type, disruption_type, severity
                            )
                            logger.info(
                                f"Simulated {disruption_type} at {node_id} (Type: {entity_type}) with severity {severity:.2f}"
                            )

            # Also check routes for disruption (e.g. traffic jams)
            for u, v, data in self.supply_chain_graph.edges(data=True):
                route_key = f"{u}-{v}"
                # Mock a traffic jam disruption on a route
                if (
                    random.random()
                    < self.sim_config["disruption_probability_base"] * 0.5
                ):  # Lower prob for traffic
                    if route_key not in self.active_disruptions:
                        disruption_type = "major_traffic_jam"
                        severity = random.uniform(0.3, 0.8)  # Traffic severity
                        self._apply_disruption_impact(
                            route_key, "route", disruption_type, severity
                        )
                        logger.info(
                            f"Simulated {disruption_type} on route {route_key} with severity {severity:.2f}"
                        )

            # Monitor and remove expired disruptions
            disruptions_to_remove = []
            for entity_id, details in self.active_disruptions.items():
                if datetime.datetime.utcnow() > details["estimated_end_time"]:
                    disruptions_to_remove.append(entity_id)
                    logger.info(
                        f"Disruption at {entity_id} ({details['disruption_type']}) has ended."
                    )
                    # Revert impacts (simplified for demo)
                    # This would involve storing original values and restoring them

            for entity_id in disruptions_to_remove:
                del self.active_disruptions[entity_id]

            await asyncio.sleep(self.sim_config["simulation_interval_seconds"])


if __name__ == "__main__":
    import redis

    config_file = "conf/environments/dev.yaml"
    sc_graph_file = "data_nexus/supply_chain_graph.gml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(sc_graph_file), exist_ok=True)
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
    disruption_simulator:
      enabled: true
      simulation_interval_seconds: 5 # Faster for demo
      disruption_probability_base: 0.2 # Higher for easier demo trigger
      disruption_types: ["factory_fire", "port_closure", "major_traffic_jam", "supplier_delay"]
      impact_factors:
        factory_fire: {"stock_reduction_percent": 0.5, "lead_time_increase_days": 10}
        port_closure: {"route_capacity_reduction_percent": 0.8, "travel_time_increase_hours": 48}
        major_traffic_jam: {"route_travel_time_multiplier": 2.0}
        supplier_delay: {"lead_time_increase_days": 7, "production_rate_reduction_percent": 0.2}
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
        print("Connected to Redis. Populating dummy data for DisruptionSimulator.")

        # Populate Redis with data mirroring graph entities
        r.set(
            "product:PROD_A",
            json.dumps(
                {
                    "current_stock": 100,
                    "production_rate": 10,
                    "lead_time_days": 5,
                    "criticality": "high",
                }
            ),
        )
        r.set(
            "warehouse:WH_001",
            json.dumps(
                {
                    "location": {"lat": 35.7, "lon": 51.4},
                    "capacity": 1000,
                    "current_occupancy": 500,
                }
            ),
        )
        r.set(
            "supplier:SUP_001",
            json.dumps(
                {
                    "location": {"lat": 34.0, "lon": 50.0},
                    "product_types": ["PROD_A"],
                    "reliability": 0.9,
                }
            ),
        )
        r.set(
            "port:PORT_001",
            json.dumps(
                {"location": {"lat": 36.0, "lon": 51.5}, "traffic_capacity": 500}
            ),
        )
        r.set(
            "manufacturer:MFG_001",
            json.dumps(
                {"location": {"lat": 34.5, "lon": 50.5}, "production_capacity": 200}
            ),
        )
        r.set(
            "route:SUP_001-MFG_001",
            json.dumps(
                {"product": "raw_material_A", "travel_time_hours": 12, "capacity": 100}
            ),
        )
        r.set(
            "route:MFG_001-WH_001",
            json.dumps({"product": "PROD_A", "travel_time_hours": 24, "capacity": 50}),
        )
        r.set(
            "route:PORT_001-WH_001",
            json.dumps({"product": "PROD_B", "travel_time_hours": 6, "capacity": 200}),
        )

    except redis.exceptions.ConnectionError:
        print("Redis not running. Disruption simulator will start with empty data.")

    async def main_disruption_simulator():
        simulator = DisruptionSimulator(config_file)
        print("Starting DisruptionSimulator for 30 seconds...")
        try:
            await asyncio.wait_for(simulator.simulate_disruptions_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("\nDisruptionSimulator demo timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nDisruptionSimulator demo stopped by user.")

    asyncio.run(main_disruption_simulator())
