import random
import redis
import json
import datetime
import yaml
import os
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeTelemetryAggregator:
    def __init__(
        self,
        config_path="conf/environments/prod.yaml",
        environment="dev",
        aggregation_window_seconds=60,
    ):
        self.config = self._load_config(config_path)
        self.redis_config = self.config["environments"][environment]["redis"]
        self.aggregation_window_seconds = aggregation_window_seconds

        self.client = redis.StrictRedis(
            host=self.redis_config["host"],
            port=self.redis_config["port"],
            db=self.redis_config["db"],
            decode_responses=True,
        )
        self.telemetry_stream_key = self.config["environments"][environment][
            "kafka"
        ].get(
            "topic_telemetry_data", "dev_telemetry_events"
        )  # Using Kafka topic name as Redis stream name
        self.consumer_group = "telemetry_aggregator_group"
        self.consumer_name = f"aggregator_instance_{os.getpid()}"

        self._initialize_stream_consumer_group()

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
    kafka: # Placeholder for config structure reference
      topic_telemetry_data: dev_telemetry_stream
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _initialize_stream_consumer_group(self):
        try:
            self.client.xgroup_create(
                self.telemetry_stream_key, self.consumer_group, "0", mkstream=True
            )
            logger.info(
                f"Redis Stream Consumer Group '{self.consumer_group}' created for '{self.telemetry_stream_key}'."
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(
                    f"Redis Stream Consumer Group '{self.consumer_group}' already exists."
                )
            else:
                logger.error(f"Error creating Redis Stream Consumer Group: {e}")
                raise

    def process_telemetry_batch(self, messages: list):
        if not messages:
            return

        driver_telemetry_data = defaultdict(lambda: defaultdict(list))
        current_time = datetime.datetime.utcnow()

        for msg_id, data in messages:
            driver_id = data.get("driver_id")
            timestamp_str = data.get("timestamp")

            if not driver_id or not timestamp_str:
                logger.warning(f"Skipping malformed telemetry message: {data}")
                continue

            try:
                msg_timestamp = datetime.datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
            except ValueError:
                logger.warning(
                    f"Skipping telemetry message with invalid timestamp format: {timestamp_str}"
                )
                continue

            # Filter messages within the aggregation window
            if (
                current_time - msg_timestamp
            ).total_seconds() <= self.aggregation_window_seconds:
                for key, value in data.items():
                    if key not in ["driver_id", "timestamp"]:
                        try:
                            driver_telemetry_data[driver_id][key].append(float(value))
                        except (ValueError, TypeError):
                            logger.debug(
                                f"Non-numeric value for {key}: {value}, skipping aggregation."
                            )

        aggregated_metrics = {}
        for driver_id, metrics in driver_telemetry_data.items():
            agg_data = {
                "timestamp_utc": current_time.isoformat(),
                "last_seen_node_id": metrics.get("current_node_id", [None])[
                    -1
                ],  # Last reported node
            }
            for metric_name, values in metrics.items():
                if values:  # Only aggregate if values exist
                    agg_data[f"avg_{metric_name}"] = round(sum(values) / len(values), 2)
                    agg_data[f"min_{metric_name}"] = round(min(values), 2)
                    agg_data[f"max_{metric_name}"] = round(max(values), 2)

            # Persist aggregated data to Redis (e.g., as a HASH or JSON string in a KEY)
            # Example key: aggregated_driver_telemetry:driver_id
            self.client.set(
                f"aggregated_driver_telemetry:{driver_id}", json.dumps(agg_data)
            )
            aggregated_metrics[driver_id] = agg_data
            logger.debug(f"Aggregated telemetry for {driver_id}: {agg_data}")

        return aggregated_metrics

    def consume_and_aggregate_loop(
        self, poll_interval_ms=1000, max_messages_per_read=100
    ):
        logger.info(
            f"Starting telemetry aggregator loop. Reading from '{self.telemetry_stream_key}'..."
        )
        last_id = "0-0"  # Start reading from the beginning of the stream if no history

        while True:
            try:
                # Read messages from the stream
                response = self.client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.telemetry_stream_key: last_id},
                    count=max_messages_per_read,
                    block=poll_interval_ms,
                )

                if response:
                    # response is like [[stream_name, [[msg_id, {field: val}]] ]]
                    for stream_name, messages in response:
                        parsed_messages = []
                        for msg_id, msg_data_dict in messages:
                            parsed_messages.append((msg_id, msg_data_dict))

                        if parsed_messages:
                            self.process_telemetry_batch(parsed_messages)

                            # Acknowledge messages
                            self.client.xack(
                                self.telemetry_stream_key,
                                self.consumer_group,
                                *[m[0] for m in parsed_messages],
                            )
                            last_id = parsed_messages[-1][
                                0
                            ]  # Update last_id for next read

                # Sleep briefly if no messages or for consistent interval
                # time.sleep(poll_interval_ms / 1000.0)

            except redis.exceptions.ConnectionError:
                logger.error("Redis connection lost, attempting to reconnect...")
                datetime.time.sleep(5)  # Wait before retrying
                self.client.ping()  # Reconnect attempt
            except Exception as e:
                logger.error(
                    f"Unexpected error in aggregation loop: {e}", exc_info=True
                )
                datetime.time.sleep(5)


if __name__ == "__main__":
    # Ensure Redis is running locally: `docker run --name some-redis -p 6379:6379 -d redis`
    # You also need a KafkaProducerSimulator (from previous code) to push data to dev_telemetry_stream
    # Example: python src/stream_processing/kafka_producer_simulator.py

    # Prepare dummy config if not present
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
    kafka:
      bootstrap_servers: ['localhost:9092']
      topic_telemetry_data: dev_telemetry_stream
"""
            )

    aggregator = RealtimeTelemetryAggregator(config_file)

    # Simulate some initial telemetry data being pushed to Redis Stream for testing
    print("Pushing dummy telemetry data to Redis Stream 'dev_telemetry_stream'...")
    for i in range(5):
        driver_id = f"driver_{random.randint(1,2)}"
        telemetry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "driver_id": driver_id,
            "current_node_id": random.randint(1, 100),
            "speed_kph": random.uniform(10, 80),
            "fuel_level_percent": random.uniform(20, 90),
            "load_kg": random.uniform(0, 50),
        }
        aggregator.client.xadd("dev_telemetry_stream", telemetry)
        datetime.time.sleep(0.1)
    print("Finished pushing dummy data.")

    print(
        "Starting RealtimeTelemetryAggregator loop for 30 seconds (or until Ctrl+C)..."
    )
    try:
        aggregator.consume_and_aggregate_loop(
            poll_interval_ms=500, max_messages_per_read=20
        )
    except KeyboardInterrupt:
        print("\nAggregator stopped by user.")

    print("\nChecking aggregated telemetry in Redis:")
    for key in aggregator.client.scan_iter("aggregated_driver_telemetry:*"):
        val = aggregator.client.get(key)
        print(f"  {key}: {val}")
