import kafka
import json
import time
import datetime
import random
import yaml
import os
import logging
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KafkaProducerSimulator:
    def __init__(self, config_path, environment="dev"):
        self.config = self._load_config(config_path)
        self.env_config = self.config["environments"][environment]
        self.producer = kafka.KafkaProducer(
            bootstrap_servers=self.env_config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )
        self.traffic_topic = self.env_config["kafka"]["topic_traffic_data"]
        self.order_topic = self.env_config["kafka"]["topic_order_data"]
        self.telemetry_topic = self.env_config["kafka"].get(
            "topic_telemetry_data", "dev_telemetry_events"
        )

        self.graph = self._load_graph()
        self.nodes = list(self.graph.nodes())
        self.edges = list(self.graph.edges(keys=True))

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_graph(self):
        # This assumes osm_processing_config has the graph path
        osm_config_path = "conf/osm_processing_config.yaml"
        if not os.path.exists(osm_config_path):
            raise FileNotFoundError(
                f"OSM processing config not found at {osm_config_path}"
            )
        with open(osm_config_path, "r") as f:
            osm_config = yaml.safe_load(f)
        graph_path = osm_config["graph_serialization"]["output_path"]
        if not os.path.exists(graph_path):
            logger.warning(
                f"Graph not found at {graph_path}. Generating a small dummy graph."
            )
            G_temp = nx.MultiDiGraph()
            for i in range(10):
                G_temp.add_node(
                    i, x=random.uniform(51.2, 51.7), y=random.uniform(35.5, 35.9)
                )
            for i in range(9):
                G_temp.add_edge(
                    i,
                    i + 1,
                    key=0,
                    travel_time=random.uniform(30, 120),
                    length=random.uniform(100, 500),
                )
            nx.write_gml(G_temp, graph_path)
        return nx.read_gml(graph_path)

    def generate_and_send_traffic_event(self):
        u, v, key = random.choice(self.edges)
        base_travel_time = self.graph[u][v][key].get("travel_time", 60)
        traffic_factor = random.uniform(0.5, 2.5)
        current_travel_time = max(10, base_travel_time * traffic_factor)

        event = {
            "timestamp": datetime.datetime.utcnow(),
            "u": u,
            "v": v,
            "key": key,
            "current_travel_time": current_travel_time,
        }
        self.producer.send(self.traffic_topic, event)
        logger.debug(f"Sent traffic event: {event}")

    def generate_and_send_order_event(self):
        order_id = f"ORD_{int(time.time() * 1000)}_{random.randint(0,999)}"
        origin_node = random.choice(self.nodes)
        destination_node = random.choice(self.nodes)
        while destination_node == origin_node:
            destination_node = random.choice(self.nodes)

        pickup_time = datetime.datetime.utcnow() + datetime.timedelta(
            minutes=random.randint(0, 60)
        )

        event = {
            "order_id": order_id,
            "origin_node": origin_node,
            "destination_node": destination_node,
            "weight": round(random.uniform(0.1, 15.0), 2),
            "volume": round(random.uniform(0.01, 2.0), 2),
            "pickup_time_start": pickup_time,
            "pickup_time_end": pickup_time + datetime.timedelta(minutes=15),
            "delivery_time_latest": pickup_time
            + datetime.timedelta(hours=random.randint(1, 4)),
            "status": "pending",
        }
        self.producer.send(self.order_topic, event)
        logger.debug(f"Sent order event: {event}")

    def generate_and_send_vehicle_telemetry(
        self, driver_id_prefix="driver", num_drivers=5
    ):
        driver_id = f"{driver_id_prefix}_{random.randint(1, num_drivers)}"
        current_node_id = random.choice(self.nodes)

        event = {
            "timestamp": datetime.datetime.utcnow(),
            "driver_id": driver_id,
            "current_node_id": current_node_id,
            "speed_kph": random.uniform(10, 80),
            "fuel_level_percent": random.uniform(10, 100),
            "load_kg": random.uniform(0, 20),
        }
        self.producer.send(self.telemetry_topic, event)
        logger.debug(f"Sent telemetry event: {event}")

    def run_simulation(
        self,
        duration_seconds=300,
        traffic_interval=5,
        order_interval=10,
        telemetry_interval=3,
    ):
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            if random.random() < (1.0 / traffic_interval):
                self.generate_and_send_traffic_event()
            if random.random() < (1.0 / order_interval):
                self.generate_and_send_order_event()
            if random.random() < (1.0 / telemetry_interval):
                self.generate_and_send_vehicle_telemetry()
            time.sleep(1)  # Base interval

        self.producer.flush()
        logger.info("Kafka simulation finished.")


if __name__ == "__main__":
    # Ensure config and graph paths exist for local testing
    conf_env_path = "conf/environments/dev.yaml"
    conf_osm_path = "conf/osm_processing_config.yaml"
    os.makedirs(os.path.dirname(conf_env_path), exist_ok=True)
    os.makedirs(os.path.dirname(conf_osm_path), exist_ok=True)

    if not os.path.exists(conf_env_path):
        with open(conf_env_path, "w") as f:
            f.write(
                """
environment: development
redis:
  host: localhost
  port: 6379
  db: 0
kafka:
  bootstrap_servers: ['localhost:9092']
  topic_traffic_data: dev_traffic_events
  topic_order_data: dev_order_events
  topic_telemetry_data: dev_telemetry_events
"""
            )
    if not os.path.exists(conf_osm_path):
        with open(conf_osm_path, "w") as f:
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

    # To run this, ensure Kafka is running on localhost:9092
    # e.g., with `docker-compose -f deployment_ops/docker-compose.yaml up -d kafka`
    producer_sim = KafkaProducerSimulator(conf_env_path)
    logger.info("Starting Kafka producer simulation for 1 minute.")
    producer_sim.run_simulation(duration_seconds=60)
