import random
import prometheus_client
import time
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    _instance = None

    def __new__(cls, config_path="conf/environments/prod.yaml", environment="dev"):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        if self._initialized:
            return

        self.config = self._load_config(config_path)
        self.metrics_config = self.config["environments"][environment][
            "monitoring"
        ].get("prometheus", {})
        self.port = self.metrics_config.get("port", 8000)
        self.enabled = self.metrics_config.get("enabled", False)
        self.job_name = self.metrics_config.get(
            "job_name", "logistics_optimization_service"
        )

        if self.enabled:
            try:
                prometheus_client.start_http_server(self.port)
                logger.info(f"Prometheus metrics server started on port {self.port}")
                self._register_common_metrics()
            except OSError as e:
                logger.error(
                    f"Failed to start Prometheus HTTP server on port {self.port}: {e}. Metrics will be disabled."
                )
                self.enabled = False
        else:
            logger.info("Prometheus metrics collection is disabled.")

        self._initialized = True

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    monitoring:
      prometheus:
        enabled: true
        port: 8000
        job_name: logistics_dev_service
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _register_common_metrics(self):
        # API Request Counters
        self.api_requests_total = prometheus_client.Counter(
            "http_requests_total",
            "Total HTTP Requests",
            ["method", "endpoint", "status"],
        )
        self.api_request_duration_seconds = prometheus_client.Histogram(
            "http_request_duration_seconds",
            "HTTP Request duration in seconds",
            ["method", "endpoint"],
        )

        # Logistics Specific Metrics
        self.orders_total = prometheus_client.Gauge(
            "logistics_orders_total", "Total orders processed"
        )
        self.orders_pending_count = prometheus_client.Gauge(
            "logistics_orders_pending_count", "Number of pending orders"
        )
        self.orders_in_transit_count = prometheus_client.Gauge(
            "logistics_orders_in_transit_count", "Number of orders in transit"
        )
        self.orders_delivered_count = prometheus_client.Gauge(
            "logistics_orders_delivered_count", "Number of delivered orders"
        )

        self.driver_available_count = prometheus_client.Gauge(
            "logistics_driver_available_count", "Number of available drivers"
        )
        self.driver_on_route_count = prometheus_client.Gauge(
            "logistics_driver_on_route_count", "Number of drivers on route"
        )
        self.driver_current_load_kg = prometheus_client.Gauge(
            "logistics_driver_current_load_kg",
            "Current load of a driver in kg",
            ["driver_id"],
        )

        self.traffic_travel_time_seconds = prometheus_client.Gauge(
            "logistics_traffic_travel_time_seconds",
            "Current travel time on an edge",
            ["u", "v", "key"],
        )

        logger.info("Prometheus metrics registered.")

    def record_api_request(
        self, method: str, endpoint: str, status: int, duration: float
    ):
        if self.enabled:
            self.api_requests_total.labels(method, endpoint, status).inc()
            self.api_request_duration_seconds.labels(method, endpoint).observe(duration)

    def set_orders_metrics(
        self, total: int, pending: int, in_transit: int, delivered: int
    ):
        if self.enabled:
            self.orders_total.set(total)
            self.orders_pending_count.set(pending)
            self.orders_in_transit_count.set(in_transit)
            self.orders_delivered_count.set(delivered)

    def set_driver_metrics(
        self, available_count: int, on_route_count: int, driver_loads: dict
    ):
        if self.enabled:
            self.driver_available_count.set(available_count)
            self.driver_on_route_count.set(on_route_count)
            for driver_id, load in driver_loads.items():
                self.driver_current_load_kg.labels(driver_id).set(load)

    def set_traffic_metrics(self, u: int, v: int, key: int, travel_time: float):
        if self.enabled:
            self.traffic_travel_time_seconds.labels(u, v, key).set(travel_time)


if __name__ == "__main__":
    # Ensure config file exists for dev environment
    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    monitoring:
      prometheus:
        enabled: true
        port: 8000
        job_name: logistics_dev_service
"""
            )

    # Initialize the MetricsCollector (starts HTTP server if enabled)
    metrics = MetricsCollector(config_file)

    if metrics.enabled:
        print(
            f"Prometheus metrics available at http://localhost:{metrics.port}/metrics"
        )
        print("Sending dummy metrics for 30 seconds...")

        # Simulate some activity
        for i in range(5):
            # API Metrics
            metrics.record_api_request(
                "POST", "/optimize", 200, random.uniform(0.1, 0.5)
            )
            metrics.record_api_request(
                "GET", "/status", 200, random.uniform(0.01, 0.05)
            )
            metrics.record_api_request("POST", "/route", 500, random.uniform(1.0, 2.0))

            # Order Metrics
            metrics.set_orders_metrics(
                total=100 + i * 10,
                pending=20 - i,
                in_transit=15 + i * 2,
                delivered=65 + i * 9,
            )

            # Driver Metrics
            driver_loads = {
                "driver_1": random.uniform(0, 10),
                "driver_2": random.uniform(0, 5),
                "driver_3": random.uniform(0, 15),
            }
            metrics.set_driver_metrics(
                available_count=3 - i % 2,
                on_route_count=2 + i % 2,
                driver_loads=driver_loads,
            )

            # Traffic Metrics
            metrics.set_traffic_metrics(100, 101, 0, random.uniform(50, 150))
            metrics.set_traffic_metrics(200, 201, 0, random.uniform(30, 90))

            time.sleep(5)
        print("Finished sending dummy metrics.")
    else:
        print("Metrics collector is disabled, no metrics were exposed.")
