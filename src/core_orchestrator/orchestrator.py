import asyncio
import datetime
import random
import yaml
import os
import logging
import json
from typing import List

from ..data_nexus.database_manager import DatabaseManager
from ..data_nexus.realtime_telemetry_aggregator import RealtimeTelemetryAggregator
from ..stream_processing.spark_kafka_consumer import SparkKafkaConsumer
try:
    from .agent_manager import AgentManager
except (ImportError, RuntimeError):
    AgentManager = None
from ..routing.astar_optimization_logic import AStarRouting
from ..feature_forge.feature_store_client import FeatureStoreClient
from ..monitoring.alert_manager import AlertManager
from ..utils.metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.env_config = self.config["environments"][environment]
        self.deployment_environment = environment

        # Initialize core components
        self.db_manager = DatabaseManager(config_path, environment)
        self.telemetry_aggregator = RealtimeTelemetryAggregator(
            config_path, environment
        )
        self.spark_consumer = SparkKafkaConsumer(
            config_path, environment
        )  # Handles Kafka data ingestion
        try:
            self.agent_manager = AgentManager(
                config_path, environment
            )  # Manages RL agent interactions
        except Exception as e:
            logger.warning(f"AgentManager initialization failed: {e}. RL features will be disabled.")
            self.agent_manager = None
        self.feature_store_client = FeatureStoreClient(config_path, environment)
        self.alert_manager = AlertManager(config_path, environment)
        self.metrics_collector = MetricsCollector(config_path, environment)

        self.router = self._initialize_router()
        self.current_simulation_time = (
            datetime.datetime.utcnow()
        )  # Represents actual system time

        self._active_drivers = {}  # Cache for active driver states
        self._pending_orders = {}  # Cache for pending orders

        logger.info("Orchestrator initialized. Connecting to database...")
        try:
            self.db_manager.connect()
            self.db_manager.create_tables()
            logger.info("Orchestrator ready to run.")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}. Database features will be disabled.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    database:
      host: localhost
      port: 5432
      user: dev_user
      password: dev_password
      dbname: logistics_dev
    redis:
      host: localhost
      port: 6379
      db: 0
    kafka:
      bootstrap_servers: ['localhost:9092']
      topic_traffic_data: dev_traffic_events
      topic_order_data: dev_order_events
      topic_telemetry_data: dev_telemetry_stream
    rl_agent:
      inference_endpoint: http://localhost:8001/v2/models/rl_agent_model/infer
      model_version: dev_v1.0
      explore_probability: 0.05
      gnn_model_path: rl_model_registry/gcn_model.pth
      rl_checkpoint_path: rl_model_registry/dev_v1.0/checkpoint_000100
    osm_processing_config_path: conf/osm_processing_config.yaml
    rl_agent_params_config_path: conf/rl_agent_params.yaml
    routing_config_path: conf/routing_engine_config.yaml
    alerting:
      slack:
        enabled: false
      email:
        enabled: false
      rules:
        - name: HighPendingOrders
          metric: num_pending_orders
          threshold: 50
          operator: ">"
          window_minutes: 5
          severity: warning
          cooldown_minutes: 10
          channels: ["slack"]
        - name: CriticalDeliveryTimeAnomaly
          metric: avg_delivery_time_minutes
          threshold: 2.5
          operator: "z_score_gt"
          window_minutes: 1
          severity: critical
          cooldown_minutes: 5
          channels: ["slack", "email"]
        - name: DriverIdleAlert
          metric: driver_idle_percentage
          threshold: 15
          operator: ">"
          window_minutes: 10
          severity: info
          cooldown_minutes: 30
          channels: ["slack"]
    monitoring:
      prometheus:
        enabled: true
        port: 8000
        job_name: logistics_dev_service
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _initialize_router(self):
        # The OSMNxProcessor handles graph loading/generation based on its config
        osm_processor = (
            self.agent_manager.osm_processor
        )  # Reuse the one from agent_manager
        graph = osm_processor.get_graph()
        routing_config_path = self.env_config.get(
            "routing_config_path", "conf/routing_engine_config.yaml"
        )
        return AStarRouting(graph, routing_config_path)

    async def _update_system_state(self):
        """
        Periodically updates the cached system state (drivers, orders, traffic)
        from the Feature Store (Redis) and database.
        """
        # Fetch latest driver telemetry from Redis (aggregated)
        driver_telemetry_keys = self.feature_store_client.client.keys(
            "aggregated_driver_telemetry:*"
        )
        driver_ids = [
            k.split("aggregated_driver_telemetry:")[1] for k in driver_telemetry_keys
        ]

        aggregated_telemetry = {}
        if driver_ids:
            raw_telemetry_data = self.feature_store_client.client.mget(
                driver_telemetry_keys
            )
            for i, data in enumerate(raw_telemetry_data):
                if data:
                    agg_id = driver_ids[i]
                    aggregated_telemetry[agg_id] = json.loads(data)

        # Fetch all drivers from DB and merge with telemetry
        db_drivers = self.db_manager._execute_query(
            "SELECT * FROM drivers;", fetch=True
        )
        self._active_drivers = {}
        for drv_row in db_drivers:
            driver_data = dict(drv_row)
            driver_id = driver_data["driver_id"]
            if driver_id in aggregated_telemetry:
                # Merge telemetry data, prioritizing latest telemetry for location/speed
                merged_data = {**driver_data, **aggregated_telemetry[driver_id]}
                # Ensure node_id is consistent
                merged_data["current_location_node_id"] = merged_data.get(
                    "last_seen_node_id", driver_data["current_location_node_id"]
                )
            else:
                merged_data = driver_data
            self._active_drivers[driver_id] = merged_data

        # Fetch pending orders from DB (or Redis if orders are also cached there)
        db_orders = self.db_manager._execute_query(
            "SELECT * FROM orders WHERE status = 'pending';", fetch=True
        )
        self._pending_orders = {o["order_id"]: dict(o) for o in db_orders}

        # Update real-time traffic context in router and agent manager
        traffic_keys = self.feature_store_client.client.keys("traffic:edge:*")
        if traffic_keys:
            traffic_data = self.feature_store_client.client.mget(traffic_keys)
            # Update graph in router and agent_manager based on latest traffic
            updated_graph = (
                self.agent_manager.current_graph.copy()
            )  # Start with current graph
            for key_str, traffic_json in zip(traffic_keys, traffic_data):
                if traffic_json:
                    traffic_info = json.loads(traffic_json)
                    edge_parts = key_str.split(":")[1].split("-")
                    u, v, k = int(edge_parts[0]), int(edge_parts[1]), int(edge_parts[2])

                    if updated_graph.has_edge(u, v, key=k):
                        base_travel_time = updated_graph[u][v][k].get(
                            "travel_time", traffic_info["current_travel_time"]
                        )  # Fallback if no base
                        traffic_factor = (
                            traffic_info["current_travel_time"] / base_travel_time
                            if base_travel_time > 0
                            else 1.0
                        )
                        updated_graph[u][v][k]["traffic_factor"] = traffic_factor
                        updated_graph[u][v][k]["current_predicted_travel_time"] = (
                            traffic_info["current_travel_time"]
                        )
                        # The router's graph must also reflect this
                        self.router.update_edge_traffic(u, v, k, traffic_factor)
                    else:
                        logger.warning(
                            f"Traffic update for non-existent edge: {u}-{v}-{k}"
                        )

            # AgentManager also needs updated graph context for GNN embeddings
            await self.agent_manager.update_graph_context(updated_graph)

        logger.debug(
            f"System state updated. Drivers: {len(self._active_drivers)}, Pending Orders: {len(self._pending_orders)}"
        )

    async def _trigger_rl_actions(self):
        """
        Iterates through active drivers and requests actions from the RL Agent Manager.
        """
        tasks = []
        for driver_id, driver_state in self._active_drivers.items():
            if (
                driver_state["status"] == "available"
                or driver_state["status"] == "idle"
            ):
                # Determine available actions for the driver
                # This logic is crucial and depends on the specific environment/tasks
                available_actions = self._determine_available_actions(
                    driver_id, driver_state
                )

                tasks.append(
                    self.agent_manager.get_agent_action(
                        driver_id, driver_state, available_actions
                    )
                )

        if tasks:
            results = await asyncio.gather(*tasks)
            for i, driver_id in enumerate(self._active_drivers.keys()):
                action = results[i]
                if action:
                    logger.info(f"Driver {driver_id} received action: {action}")
                    await self._execute_action(driver_id, action)
        else:
            logger.info("No available drivers to trigger RL actions for.")

    def _determine_available_actions(
        self, driver_id: str, driver_state: dict
    ) -> List[str]:
        """
        Logic to determine what actions a driver can realistically take.
        Example: "move_to_node_X", "pickup_order_Y", "deliver_order_Z", "wait", "refuel"
        """
        actions = []
        if driver_state["status"] == "available" or driver_state["status"] == "idle":
            # Can move to any nearby node (simplified to a few random ones for demo)
            current_node = driver_state.get("current_location_node_id")
            if current_node and current_node in self.agent_manager.current_graph:
                neighbors = list(
                    self.agent_manager.current_graph.neighbors(current_node)
                )
                if neighbors:
                    for _ in range(
                        min(3, len(neighbors))
                    ):  # Offer up to 3 random neighbor moves
                        actions.append(f"move_to_node_{random.choice(neighbors)}")

            # Can pick up pending orders at current location
            for order_id, order in self._pending_orders.items():
                if (
                    order["origin_node"] == current_node
                    and order["status"] == "pending"
                ):
                    actions.append(f"pickup_order_{order_id}")

            # Can deliver orders currently carried
            assigned_orders = driver_state.get("assigned_orders", [])
            for order_id in assigned_orders:
                if (
                    order_id in self._pending_orders
                ):  # If order is still pending/picked_up
                    order = self._pending_orders[order_id]
                    if (
                        order["status"] == "picked_up"
                        and order["destination_node"] == current_node
                    ):
                        actions.append(f"deliver_order_{order_id}")

            actions.append("wait")

        # Add a default action if none are dynamic
        if not actions:
            actions.append("wait")
        return actions

    async def _execute_action(self, driver_id: str, action: str):
        """
        Executes the chosen action for a driver and updates the system state.
        This would involve interacting with the routing engine, updating database, etc.
        """
        driver = self._active_drivers.get(driver_id)
        if not driver:
            logger.error(
                f"Attempted to execute action for non-existent driver: {driver_id}"
            )
            return

        current_node_id = driver["current_location_node_id"]
        self.agent_manager.current_graph.nodes[current_node_id]["y"]
        self.agent_manager.current_graph.nodes[current_node_id]["x"]

        if action.startswith("move_to_node_"):
            target_node = int(action.split("_")[-1])
            logger.info(
                f"Driver {driver_id} moving from {current_node_id} to {target_node}."
            )
            # Use routing engine
            route_info = self.router.get_shortest_path(current_node_id, target_node)
            if route_info:
                # Update driver's state to 'on_route', store route, estimated time
                driver["status"] = "on_route"
                driver["current_route"] = route_info["path"]
                driver["estimated_travel_time_seconds"] = route_info["travel_time"]
                driver["target_node"] = (
                    target_node  # Set target node for movement simulation
                )
                self.db_manager.insert_driver(driver)  # Persist status change
                logger.info(
                    f"Driver {driver_id} routed to {target_node} ({route_info['travel_time']:.2f}s)."
                )
            else:
                logger.warning(
                    f"Could not find route for {driver_id} from {current_node_id} to {target_node}. Driver waits."
                )
                driver["status"] = "idle"
                self.db_manager.insert_driver(driver)  # Persist status change

        elif action.startswith("pickup_order_"):
            order_id = action.split("_")[-1]
            if (
                order_id in self._pending_orders
                and self._pending_orders[order_id]["origin_node"] == current_node_id
            ):
                order = self._pending_orders[order_id]
                order["status"] = "picked_up"
                order["actual_pickup_time"] = datetime.datetime.utcnow()
                driver["current_load"] += order["weight"]
                driver["assigned_orders"].append(order_id)
                self.db_manager.insert_order(order)
                self.db_manager.insert_driver(driver)
                logger.info(f"Driver {driver_id} picked up order {order_id}.")
            else:
                logger.warning(
                    f"Driver {driver_id} tried to pick up {order_id} but it's not available or at current location."
                )

        elif action.startswith("deliver_order_"):
            order_id = action.split("_")[-1]
            if (
                order_id in self._pending_orders
                and self._pending_orders[order_id]["destination_node"]
                == current_node_id
                and order_id in driver["assigned_orders"]
            ):
                order = self._pending_orders[order_id]
                order["status"] = "delivered"
                order["actual_delivery_time"] = datetime.datetime.utcnow()
                driver["current_load"] -= order["weight"]
                driver["assigned_orders"].remove(order_id)
                self.db_manager.insert_order(order)
                self.db_manager.insert_driver(driver)
                logger.info(f"Driver {driver_id} delivered order {order_id}.")
            else:
                logger.warning(
                    f"Driver {driver_id} tried to deliver {order_id} but conditions not met."
                )

        elif action == "wait":
            logger.info(f"Driver {driver_id} is waiting at node {current_node_id}.")
            driver["status"] = "idle"
            self.db_manager.insert_driver(driver)

        # Update metrics after action
        self._update_prometheus_metrics()

    def _update_prometheus_metrics(self):
        """Pushes current system state metrics to Prometheus."""
        if not self.metrics_collector.enabled:
            return

        total_orders = len(self._pending_orders) + len(
            self.db_manager._execute_query(
                "SELECT * FROM orders WHERE status = 'delivered'", fetch=True
            )
        )
        pending_orders_count = sum(
            1 for o in self._pending_orders.values() if o["status"] == "pending"
        )
        in_transit_orders_count = sum(
            1 for o in self._pending_orders.values() if o["status"] == "picked_up"
        )  # Orders picked up but not delivered
        delivered_orders_count = len(
            self.db_manager._execute_query(
                "SELECT * FROM orders WHERE status = 'delivered'", fetch=True
            )
        )

        self.metrics_collector.set_orders_metrics(
            total=total_orders,
            pending=pending_orders_count,
            in_transit=in_transit_orders_count,
            delivered=delivered_orders_count,
        )

        available_drivers = sum(
            1
            for d in self._active_drivers.values()
            if d["status"] == "available" or d["status"] == "idle"
        )
        on_route_drivers = sum(
            1
            for d in self._active_drivers.values()
            if d["status"] == "on_route" or d["status"] == "traveling"
        )
        driver_loads = {
            did: d.get("current_load", 0) for did, d in self._active_drivers.items()
        }

        self.metrics_collector.set_driver_metrics(
            available_count=available_drivers,
            on_route_count=on_route_drivers,
            driver_loads=driver_loads,
        )

        # Traffic metrics would be set by a stream processor, not here directly
        # Example: self.metrics_collector.set_traffic_metrics(u, v, k, travel_time)

    async def run_main_loop(
        self, update_interval_seconds=10, rl_decision_interval_seconds=30
    ):
        """
        Main orchestration loop.
        - Ingests data (via Spark consumer in background).
        - Aggregates telemetry.
        - Updates system state from DB/Feature Store.
        - Triggers RL agent decisions for drivers.
        - Evaluates alerts.
        - Pushes metrics to monitoring.
        """
        logger.info("Starting Orchestrator main loop...")

        # Start Kafka consumers in background (Spark)
        traffic_query = self.spark_consumer.process_traffic_data()
        order_query = self.spark_consumer.process_order_data()
        # Telemetry stream is consumed by RealtimeTelemetryAggregator directly (Redis Streams)

        # Start telemetry aggregation in background
        telemetry_agg_task = asyncio.create_task(
            self.telemetry_aggregator.consume_and_aggregate_loop(poll_interval_ms=500)
        )

        last_rl_decision_time = datetime.datetime.utcnow()
        last_metric_collection_time = datetime.datetime.utcnow()

        try:
            while True:
                self.current_simulation_time = datetime.datetime.utcnow()
                logger.info(
                    f"Orchestrator loop running at {self.current_simulation_time.isoformat()}"
                )

                await self._update_system_state()

                # Check for RL decision interval
                if (
                    self.current_simulation_time - last_rl_decision_time
                ).total_seconds() >= rl_decision_interval_seconds:
                    await self._trigger_rl_actions()
                    last_rl_decision_time = self.current_simulation_time

                # Check for metric collection and alerting interval (can be more frequent than RL)
                if (
                    self.current_simulation_time - last_metric_collection_time
                ).total_seconds() >= update_interval_seconds:
                    self._update_prometheus_metrics()
                    # Placeholder: Fetch metrics for alerting (e.g., from Prometheus or direct computation)
                    # For demo, let's use some dummy metrics to check rules
                    dummy_current_metrics = {
                        "num_pending_orders": len(self._pending_orders),
                        "avg_delivery_time_minutes": 35.0,  # Placeholder
                        "driver_idle_percentage": 10.0,  # Placeholder
                    }
                    self.alert_manager.process_metrics_for_alerts(dummy_current_metrics)
                    last_metric_collection_time = self.current_simulation_time

                await asyncio.sleep(update_interval_seconds)  # Wait for next cycle

        except asyncio.CancelledError:
            logger.info("Orchestrator main loop cancelled.")
        except Exception as e:
            logger.critical(
                f"Orchestrator main loop encountered a critical error: {e}",
                exc_info=True,
            )
        finally:
            logger.info("Stopping background tasks and closing connections...")
            traffic_query.stop()
            order_query.stop()
            telemetry_agg_task.cancel()
            await telemetry_agg_task  # Ensure cancellation is awaited
            self.spark_consumer.spark.stop()
            self.db_manager.close()
            logger.info("Orchestrator gracefully shut down.")


if __name__ == "__main__":
    # Ensure all necessary config files and directories exist for the orchestrator to start
    conf_dir = "conf/environments"
    os.makedirs(conf_dir, exist_ok=True)
    dev_config_path = os.path.join(conf_dir, "dev.yaml")

    # Create dummy environment config (the Orchestrator's __init__ will create it if missing)
    # The important part is that the dummy configs for Redis, Kafka, RL Agent, OSM, Routing, Alerting, Monitoring
    # should be consistent with what Orchestrator expects.

    # Run a simple check to see if dummy configs exist. If not, the respective classes will create them.
    # To run this full stack locally:
    # 1. Ensure Docker and Docker Compose are installed.
    # 2. Go to `deployment_ops/docker/` and build the docker images: `docker build -t rl-agent-service -f Dockerfile.rl_agent_service .`
    #    `docker build -t routing-service -f Dockerfile.routing_service .`
    #    `docker build -t kafka-producer-sim -f Dockerfile.kafka_producer_sim .` (needs `kafka_producer_sim.py`)
    #    `docker build -t spark-kafka-consumer -f Dockerfile.spark_kafka_consumer .` (needs `spark_kafka_consumer.py`)
    # 3. In `deployment_ops/docker-compose.yaml`, ensure image names match the built ones.
    # 4. Make sure `graphhopper_data` contains a valid PBF for GraphHopper.
    # 5. Run `docker-compose -f deployment_ops/docker-compose.yaml up -d`
    # 6. You might need to manually create `rl_model_registry` directory and dummy files as per `agent_manager.py`'s `if __name__` block.
    # 7. Then run this file.

    async def main():
        orchestrator = Orchestrator(dev_config_path)
        try:
            await orchestrator.run_main_loop(
                update_interval_seconds=5, rl_decision_interval_seconds=15
            )
        except KeyboardInterrupt:
            logger.info("Orchestrator stopped manually.")
        finally:
            # Explicitly ensure all tasks are cleaned up
            pass

    asyncio.run(main())
