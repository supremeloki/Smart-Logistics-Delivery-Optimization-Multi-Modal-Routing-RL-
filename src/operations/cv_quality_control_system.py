import datetime
import json
import yaml
import os
import logging
import random
import asyncio
from collections import deque
from typing import Dict, Any, List
import redis
import numpy as np


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "package_in_transit:PKG_123",
            json.dumps(
                {
                    "item_id": "ITEM_A",
                    "current_status": "in_warehouse",
                    "next_scan_location": "QC_LINE_1",
                }
            ),
        )
        self.client.set(
            "warehouse_area_status:QC_LINE_1",
            json.dumps({"camera_active": True, "throughput_rate_per_min": 10}),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))


class MockComputerVisionModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def detect_defects(self, image_data: np.ndarray) -> List[Dict[str, Any]]:
        if random.random() < 0.1:  # 10% chance of detecting a defect
            defect_type = random.choice(["dent", "scratch", "mislabel", "wrong_item"])
            severity = random.uniform(0.5, 0.9)
            return [{"type": defect_type, "severity": severity, "confidence": 0.95}]
        return []

    def read_label(self, image_data: np.ndarray) -> str:
        if random.random() < 0.05:
            return "MISLABELED_ITEM_XYZ"
        return f"ITEM_{random.randint(100, 999)}"


class MockAlertManager:
    def __init__(self, config_path, environment):
        pass

    def _send_notifications(self, alert_details: Dict[str, Any], channels: List[str]):
        logging.warning(
            f"Mock Alert Sent: {alert_details['description']} (Severity: {alert_details['severity']}) via {channels}"
        )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVQualityControlSystem:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.qc_config = self.config["environments"][environment]["cv_quality_control"]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.alert_manager = MockAlertManager(config_path, environment)
        self.cv_model_defect = MockComputerVisionModel("package_defect_detector")
        self.cv_model_ocr = MockComputerVisionModel("label_reader_ocr")

        self.last_scan_time = {}  # item_id -> timestamp for debouncing
        self.detected_anomalies = deque(maxlen=100)
        logger.info("CVQualityControlSystem initialized.")

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
    cv_quality_control:
      enabled: true
      scan_interval_seconds: 2
      defect_threshold_severity: 0.6
      mislabeled_alert_threshold: 0.01 # Probability of mislabel leading to alert
      max_scan_frequency_per_item_seconds: 10
      quality_check_points: ["QC_LINE_1", "PACKING_STATION_A"]
      anomaly_handling_strategies:
        dent: "repack"
        scratch: "repack"
        mislabel: "relabel"
        wrong_item: "divert_to_returns"
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

    async def _simulate_image_capture(self, location_id: str) -> np.ndarray:
        """Simulates capturing an image from a camera at a QC point."""
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate capture latency
        return np.random.rand(1080, 1920, 3) * 255  # Mock image data

    async def perform_quality_check(self, item_id: str, current_location_id: str):
        """
        Performs real-time quality control checks on an item using computer vision.
        """
        if not self.qc_config["enabled"]:
            logger.debug(f"CV Quality Control disabled. Skipping check for {item_id}.")
            return

        last_scan_time = self.last_scan_time.get(item_id, datetime.datetime.min)
        if (
            datetime.datetime.utcnow() - last_scan_time
        ).total_seconds() < self.qc_config["max_scan_frequency_per_item_seconds"]:
            logger.debug(f"Skipping frequent scan for {item_id}.")
            return

        logger.info(f"Performing QC check for {item_id} at {current_location_id}...")
        image_data = await self._simulate_image_capture(current_location_id)

        # 1. Defect Detection
        defects = self.cv_model_defect.detect_defects(image_data)

        for defect in defects:
            if defect["severity"] >= self.qc_config["defect_threshold_severity"]:
                alert_description = f"Defect '{defect['type']}' detected on item {item_id} at {current_location_id} with severity {defect['severity']:.2f}."
                logger.warning(alert_description)
                self.alert_manager._send_notifications(
                    {
                        "item_id": item_id,
                        "location": current_location_id,
                        "defect_type": defect["type"],
                        "severity": defect["severity"],
                        "description": alert_description,
                    },
                    ["slack", "email"],
                )
                self.detected_anomalies.append(
                    {
                        "item_id": item_id,
                        "type": "defect",
                        "details": defect,
                        "timestamp": datetime.datetime.utcnow(),
                    }
                )

                # Trigger automated handling based on strategy
                handling_strategy = self.qc_config["anomaly_handling_strategies"].get(
                    defect["type"], "manual_review"
                )
                logger.info(
                    f"Automated handling for {item_id} defect: {handling_strategy}."
                )
                # In a real system, this would trigger an action (e.g., divert to repack station)
                # self.feature_store.set_feature("warehouse_action", f"ACTION_{item_id}", {"type": handling_strategy, "item_id": item_id, "triggered_by": "CV_QC"})

        # 2. Label Verification (OCR)
        scanned_label = self.cv_model_ocr.read_label(image_data)
        expected_item_id = self.feature_store.get_feature(
            "package_in_transit", item_id
        ).get("item_id")

        if expected_item_id and scanned_label != expected_item_id:
            alert_description = f"Mislabeled item {item_id} at {current_location_id}. Scanned: '{scanned_label}', Expected: '{expected_item_id}'."
            logger.error(alert_description)
            self.alert_manager._send_notifications(
                {
                    "item_id": item_id,
                    "location": current_location_id,
                    "scanned_label": scanned_label,
                    "expected_label": expected_item_id,
                    "description": alert_description,
                    "severity": "critical",
                },
                ["slack", "email"],
            )
            self.detected_anomalies.append(
                {
                    "item_id": item_id,
                    "type": "mislabeled",
                    "details": {"scanned": scanned_label, "expected": expected_item_id},
                    "timestamp": datetime.datetime.utcnow(),
                }
            )
            logger.info(
                f"Automated handling for {item_id} mislabel: {self.qc_config['anomaly_handling_strategies'].get('mislabel', 'manual_review')}."
            )

        self.last_scan_time[item_id] = datetime.datetime.utcnow()

    async def run_qc_loop(self):
        """Main loop for continuous quality control scanning."""
        logger.info("Starting CV Quality Control System loop...")

        while True:
            try:
                # Iterate through configured quality check points
                for qc_point in self.qc_config["quality_check_points"]:
                    # Simulate items passing through QC point
                    # In a real system, a sensor would detect item presence and ID
                    # For demo, just pick a random item from 'in_warehouse' status
                    all_packages = self.feature_store.client.keys(
                        "package_in_transit:*"
                    )
                    if all_packages:
                        random_package_key = random.choice(all_packages)
                        item_id = random_package_key.split(":")[1]
                        package_data = self.feature_store.get_feature(
                            "package_in_transit", item_id
                        )

                        if (
                            package_data
                            and package_data.get("current_status") == "in_warehouse"
                            and package_data.get("next_scan_location") == qc_point
                        ):
                            await self.perform_quality_check(item_id, qc_point)

            except Exception as e:
                logger.error(f"Error in CV Quality Control loop: {e}", exc_info=True)

            await asyncio.sleep(self.qc_config["scan_interval_seconds"])


if __name__ == "__main__":
    import redis

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
    cv_quality_control:
      enabled: true
      scan_interval_seconds: 1 # Faster for demo
      defect_threshold_severity: 0.5 # Lower for easier trigger
      mislabeled_alert_threshold: 0.05
      max_scan_frequency_per_item_seconds: 5
      quality_check_points: ["QC_LINE_1", "PACKING_STATION_A"]
      anomaly_handling_strategies:
        dent: "repack"
        scratch: "repack"
        mislabel: "relabel"
        wrong_item: "divert_to_returns"
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
        print("Connected to Redis. Populating dummy data for CVQualityControlSystem.")

        r.set(
            "package_in_transit:PKG_123",
            json.dumps(
                {
                    "item_id": "ITEM_A_123",
                    "current_status": "in_warehouse",
                    "next_scan_location": "QC_LINE_1",
                }
            ),
        )
        r.set(
            "package_in_transit:PKG_456",
            json.dumps(
                {
                    "item_id": "ITEM_B_456",
                    "current_status": "in_warehouse",
                    "next_scan_location": "PACKING_STATION_A",
                }
            ),
        )
        r.set(
            "package_in_transit:PKG_789",
            json.dumps(
                {
                    "item_id": "ITEM_C_789",
                    "current_status": "in_warehouse",
                    "next_scan_location": "QC_LINE_1",
                }
            ),
        )
        r.set(
            "warehouse_area_status:QC_LINE_1",
            json.dumps({"camera_active": True, "throughput_rate_per_min": 10}),
        )
        r.set(
            "warehouse_area_status:PACKING_STATION_A",
            json.dumps({"camera_active": True, "throughput_rate_per_min": 5}),
        )

    except redis.exceptions.ConnectionError:
        print(
            "Redis not running. CV Quality Control System will start with empty data."
        )

    async def main_cv_qc():
        qc_system = CVQualityControlSystem(config_file)
        print("Starting CVQualityControlSystem for 30 seconds...")
        try:
            await asyncio.wait_for(qc_system.run_qc_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("\nCVQualityControlSystem demo timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nCVQualityControlSystem demo stopped by user.")

    asyncio.run(main_cv_qc())
