from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType,
)
import yaml
import logging
import os
import redis
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkKafkaConsumer:
    def __init__(self, config_path, environment="dev"):
        self.config = self._load_config(config_path)
        self.env_config = self.config["environments"][environment]
        self.kafka_bootstrap_servers = ",".join(
            self.env_config["kafka"]["bootstrap_servers"]
        )

        try:
            self.spark = (
                SparkSession.builder.appName("LogisticsStreamProcessor")
                .config(
                    "spark.jars.packages",
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1",
                )
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel("WARN")
        except Exception as e:
            logger.warning(f"Spark session creation failed: {e}. Spark features will be disabled.")
            self.spark = None

        self.redis_client = redis.StrictRedis(
            host=self.env_config["redis"]["host"],
            port=self.env_config["redis"]["port"],
            db=self.env_config["redis"]["db"],
            decode_responses=True,
        )

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_kafka_df(self, topic):
        if self.spark is None:
            raise RuntimeError("Spark session not available")
        return (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", topic)
            .option("startingOffsets", "latest")
            .load()
        )

    def process_traffic_data(self):
        if self.spark is None:
            return None
        topic = self.env_config["kafka"]["topic_traffic_data"]
        df = self._get_kafka_df(topic)

        schema = StructType(
            [
                StructField("timestamp", StringType()),
                StructField("u", IntegerType()),
                StructField("v", IntegerType()),
                StructField("key", IntegerType()),
                StructField("current_travel_time", FloatType()),
            ]
        )

        parsed_df = (
            df.selectExpr("CAST(value AS STRING) as json_value")
            .withColumn("data", from_json(col("json_value"), schema))
            .select("data.*")
            .withColumn("processed_at", current_timestamp())
        )

        # Write to Redis (real-time updates)
        def write_traffic_to_redis(batch_df, batch_id):
            batch_data = batch_df.collect()
            if batch_data:
                pipe = self.redis_client.pipeline()
                for row in batch_data:
                    edge_id = f"{row.u}-{row.v}-{row.key}"
                    data = {
                        "current_travel_time": row.current_travel_time,
                        "timestamp": row.timestamp,
                    }
                    pipe.set(f"traffic_edge:{edge_id}", json.dumps(data))
                pipe.execute()
                logger.info(
                    f"Batch {batch_id}: Processed {len(batch_data)} traffic updates to Redis."
                )

        query = (
            parsed_df.writeStream.foreachBatch(write_traffic_to_redis)
            .outputMode("update")
            .option(
                "checkpointLocation", f"data_nexus/spark_checkpoints/traffic_{topic}"
            )
            .start()
        )

        return query

    def process_order_data(self):
        if self.spark is None:
            return None
        topic = self.env_config["kafka"]["topic_order_data"]
        df = self._get_kafka_df(topic)

        schema = StructType(
            [
                StructField("order_id", StringType()),
                StructField("origin_node", IntegerType()),
                StructField("destination_node", IntegerType()),
                StructField("weight", FloatType()),
                StructField("volume", FloatType()),
                StructField("pickup_time_start", StringType()),
                StructField("pickup_time_end", StringType()),
                StructField("delivery_time_latest", StringType()),
                StructField("status", StringType()),
            ]
        )

        parsed_df = (
            df.selectExpr("CAST(value AS STRING) as json_value")
            .withColumn("data", from_json(col("json_value"), schema))
            .select("data.*")
            .withColumn("processed_at", current_timestamp())
        )

        # Write to Redis (active orders)
        def write_orders_to_redis(batch_df, batch_id):
            batch_data = batch_df.collect()
            if batch_data:
                pipe = self.redis_client.pipeline()
                for row in batch_data:
                    order_data = row.asDict()
                    pipe.set(f"order:{row.order_id}", json.dumps(order_data))
                pipe.execute()
                logger.info(
                    f"Batch {batch_id}: Processed {len(batch_data)} order updates to Redis."
                )

        query = (
            parsed_df.writeStream.foreachBatch(write_orders_to_redis)
            .outputMode("update")
            .option(
                "checkpointLocation", f"data_nexus/spark_checkpoints/orders_{topic}"
            )
            .start()
        )

        return query

    def await_termination(self, queries):
        for query in queries:
            query.awaitTermination()


if __name__ == "__main__":
    # For a full Spark Kafka demo, you need:
    # 1. A running Kafka cluster (e.g., via Docker Compose)
    # 2. Topics created (e.g., 'dev_traffic_events', 'dev_order_events')
    # 3. A way to produce messages to these topics (e.g., with kafka-python producer)

    # Ensure config files and necessary directories exist for local testing
    conf_dir = "conf/environments"
    os.makedirs(conf_dir, exist_ok=True)
    dev_config_path = os.path.join(conf_dir, "dev.yaml")
    if not os.path.exists(dev_config_path):
        with open(dev_config_path, "w") as f:
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
"""
            )
    os.makedirs("data_nexus/spark_checkpoints", exist_ok=True)

    consumer = SparkKafkaConsumer(dev_config_path)

    logger.info("Starting Spark Kafka Consumers...")
    traffic_query = consumer.process_traffic_data()
    order_query = consumer.process_order_data()

    logger.info("Spark consumers started. Waiting for data or manual termination...")
    logger.info(
        "To test, run a Kafka producer to 'dev_traffic_events' and 'dev_order_events'."
    )
    logger.info(
        "Example traffic message: {'timestamp': '2025-10-21T10:00:00', 'u': 1, 'v': 2, 'key': 0, 'current_travel_time': 120.5}"
    )
    logger.info(
        "Example order message: {'order_id': 'ORD123', 'origin_node': 10, 'destination_node': 20, 'weight': 5.0, 'volume': 0.1, 'pickup_time_start': '2025-10-21T10:00:00', 'pickup_time_end': '2025-10-21T10:15:00', 'delivery_time_latest': '2025-10-21T11:00:00', 'status': 'pending'}"
    )

    try:
        # For demonstration, let's just let it run for a short period or until manually stopped
        # In a real deployment, this would typically awaitTermination()
        # For interactive testing in non-blocking way, comment this out or use a sleep loop
        import time

        time.sleep(60)  # Run for 60 seconds

    except KeyboardInterrupt:
        logger.info("Terminating Spark queries...")
    finally:
        traffic_query.stop()
        order_query.stop()
        consumer.spark.stop()
        logger.info("Spark Kafka Consumers stopped.")

    # Verify Redis updates
    logger.info("Checking Redis for updates (last 5 entries):")
    for key in consumer.redis_client.scan_iter("traffic_edge:*"):
        logger.info(f"Traffic Edge: {key} -> {consumer.redis_client.get(key)}")
        break  # Just one
    for key in consumer.redis_client.scan_iter("order:*"):
        logger.info(f"Order: {key} -> {consumer.redis_client.get(key)}")
        break  # Just one
