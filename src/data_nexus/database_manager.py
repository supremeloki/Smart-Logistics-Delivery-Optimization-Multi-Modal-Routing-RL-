import psycopg2
import psycopg2.extras
import yaml
import os
import logging
import datetime
import pandas as pd
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, config_path, environment="dev"):
        self.config = self._load_config(config_path)
        self.db_config = self.config["environments"][environment]["database"]
        self.conn = None

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                dbname=self.db_config["dbname"],
            )
            logger.info("Database connection established.")
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed.")

    def _execute_query(self, query, params=None, fetch=False):
        if not self.conn:
            self.connect()
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            try:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                self.conn.commit()
            except psycopg2.Error as e:
                self.conn.rollback()
                logger.error(f"Database query failed: {e}")
                raise

    def create_tables(self):
        queries = [
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id VARCHAR(255) PRIMARY KEY,
                origin_node_id INTEGER NOT NULL,
                destination_node_id INTEGER NOT NULL,
                weight FLOAT,
                volume FLOAT,
                pickup_time_start TIMESTAMP,
                pickup_time_end TIMESTAMP,
                delivery_time_latest TIMESTAMP,
                actual_pickup_time TIMESTAMP,
                actual_delivery_time TIMESTAMP,
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS drivers (
                driver_id VARCHAR(255) PRIMARY KEY,
                vehicle_type VARCHAR(50),
                capacity FLOAT,
                current_location_node_id INTEGER,
                status VARCHAR(50),
                speed_mps FLOAT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS historical_routes (
                route_id SERIAL PRIMARY KEY,
                driver_id VARCHAR(255) REFERENCES drivers(driver_id),
                order_id VARCHAR(255) REFERENCES orders(order_id),
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                planned_distance_meters FLOAT,
                actual_distance_meters FLOAT,
                planned_time_seconds FLOAT,
                actual_time_seconds FLOAT,
                path_geometry TEXT, -- JSON or WKT for path points
                traffic_factors TEXT -- JSON of edge_id: factor
            );
            """,
        ]
        for query in queries:
            self._execute_query(query)
        logger.info("Tables created or already exist.")

    def insert_order(self, order_data: dict):
        query = """
        INSERT INTO orders (order_id, origin_node_id, destination_node_id, weight, volume, 
                            pickup_time_start, pickup_time_end, delivery_time_latest, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (order_id) DO UPDATE SET
            status = EXCLUDED.status,
            actual_pickup_time = EXCLUDED.actual_pickup_time,
            actual_delivery_time = EXCLUDED.actual_delivery_time,
            pickup_time_start = EXCLUDED.pickup_time_start,
            pickup_time_end = EXCLUDED.pickup_time_end,
            delivery_time_latest = EXCLUDED.delivery_time_latest;
        """
        params = (
            order_data["order_id"],
            order_data["origin_node"],
            order_data["destination_node"],
            order_data.get("weight"),
            order_data.get("volume"),
            pd.to_datetime(order_data.get("pickup_time_start")),
            pd.to_datetime(order_data.get("pickup_time_end")),
            pd.to_datetime(order_data.get("delivery_time_latest")),
            order_data.get("status", "pending"),
        )
        self._execute_query(query, params)

    def insert_driver(self, driver_data: dict):
        query = """
        INSERT INTO drivers (driver_id, vehicle_type, capacity, current_location_node_id, status, speed_mps)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (driver_id) DO UPDATE SET
            vehicle_type = EXCLUDED.vehicle_type,
            capacity = EXCLUDED.capacity,
            current_location_node_id = EXCLUDED.current_location_node_id,
            status = EXCLUDED.status,
            speed_mps = EXCLUDED.speed_mps,
            last_updated = CURRENT_TIMESTAMP;
        """
        params = (
            driver_data["driver_id"],
            driver_data.get("vehicle_type"),
            driver_data.get("capacity"),
            driver_data.get("current_node_id"),
            driver_data.get("status"),
            driver_data.get("speed_mps"),
        )
        self._execute_query(query, params)

    def insert_historical_route(self, route_data: dict):
        query = """
        INSERT INTO historical_routes (driver_id, order_id, start_time, end_time, 
                                       planned_distance_meters, actual_distance_meters, 
                                       planned_time_seconds, actual_time_seconds, path_geometry, traffic_factors)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        params = (
            route_data.get("driver_id"),
            route_data.get("order_id"),
            pd.to_datetime(route_data.get("start_time")),
            pd.to_datetime(route_data.get("end_time")),
            route_data.get("planned_distance_meters"),
            route_data.get("actual_distance_meters"),
            route_data.get("planned_time_seconds"),
            route_data.get("actual_time_seconds"),
            json.dumps(route_data.get("path_geometry")),
            json.dumps(route_data.get("traffic_factors")),
        )
        self._execute_query(query, params)

    def get_all_orders(self):
        query = "SELECT * FROM orders;"
        return self._execute_query(query, fetch=True)

    def get_drivers_by_status(self, status):
        query = "SELECT * FROM drivers WHERE status = %s;"
        return self._execute_query(query, (status,), fetch=True)


if __name__ == "__main__":
    # Ensure a dummy config exists for startup
    conf_dir = "conf/environments"
    os.makedirs(conf_dir, exist_ok=True)
    dev_config_path = os.path.join(conf_dir, "dev.yaml")
    if not os.path.exists(dev_config_path):
        with open(dev_config_path, "w") as f:
            f.write(
                """
environment: development
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
"""
            )

    # This requires a PostgreSQL database running locally named 'logistics_dev'
    # with user 'dev_user' and password 'dev_password'.
    # Example to run PostgreSQL with Docker:
    # docker run --name some-postgres -e POSTGRES_PASSWORD=dev_password -e POSTGRES_USER=dev_user -e POSTGRES_DB=logistics_dev -p 5432:5432 -d postgres

    db_manager = DatabaseManager(dev_config_path)
    try:
        db_manager.connect()
        db_manager.create_tables()

        # Insert dummy data
        test_order = {
            "order_id": "ORD_TEST_001",
            "origin_node": 100,
            "destination_node": 200,
            "weight": 5.5,
            "volume": 0.2,
            "pickup_time_start": datetime.datetime.utcnow(),
            "pickup_time_end": datetime.datetime.utcnow()
            + datetime.timedelta(minutes=15),
            "delivery_time_latest": datetime.datetime.utcnow()
            + datetime.timedelta(hours=2),
            "status": "pending",
        }
        db_manager.insert_order(test_order)
        logger.info("Inserted test order.")

        test_driver = {
            "driver_id": "DRV_TEST_001",
            "vehicle_type": "car",
            "capacity": 20.0,
            "current_node_id": 150,
            "status": "available",
            "speed_mps": 12.0,
        }
        db_manager.insert_driver(test_driver)
        logger.info("Inserted test driver.")

        test_route = {
            "driver_id": "DRV_TEST_001",
            "order_id": "ORD_TEST_001",
            "start_time": datetime.datetime.utcnow() - datetime.timedelta(minutes=30),
            "end_time": datetime.datetime.utcnow(),
            "planned_distance_meters": 5000.0,
            "actual_distance_meters": 5200.0,
            "planned_time_seconds": 300.0,
            "actual_time_seconds": 350.0,
            "path_geometry": [{"lat": 35.7, "lon": 51.4}, {"lat": 35.71, "lon": 51.41}],
            "traffic_factors": {"100-101-0": 1.2, "101-102-0": 1.0},
        }
        db_manager.insert_historical_route(test_route)
        logger.info("Inserted test historical route.")

        # Fetch data
        all_orders = db_manager.get_all_orders()
        logger.info(f"\nAll Orders ({len(all_orders)}):")
        for order in all_orders:
            logger.info(f"  {dict(order)}")

        available_drivers = db_manager.get_drivers_by_status("available")
        logger.info(f"\nAvailable Drivers ({len(available_drivers)}):")
        for driver in available_drivers:
            logger.info(f"  {dict(driver)}")

    except Exception as e:
        logger.error(f"Error during DatabaseManager example: {e}")
    finally:
        db_manager.close()
