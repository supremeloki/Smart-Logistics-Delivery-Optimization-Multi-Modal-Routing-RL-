import datetime
import json
import yaml
import os
import logging
import random
import asyncio
from collections import deque
from typing import Dict, Any
import redis
import numpy as np


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "incident_report:INC_001",
            json.dumps(
                {
                    "type": "theft",
                    "location": {"lat": 35.71, "lon": 51.41},
                    "time": (
                        datetime.datetime.utcnow() - datetime.timedelta(minutes=5)
                    ).isoformat()
                    + "Z",
                    "severity_score": 0.6,
                }
            ),
        )
        self.client.set(
            "anomaly_detection:CAMERA_123:current",
            json.dumps(
                {
                    "anomaly_score": 0.8,
                    "type": "unusual_gathering",
                    "location": {"lat": 35.72, "lon": 51.42},
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        self.client.set(
            "historical_crime_density:ZONE_X",
            json.dumps({"density_score": 0.7, "hotspot_tendency": True}),
        )
        self.client.set(
            "citizen_preference:USER_A",
            json.dumps(
                {
                    "alert_radius_km": 2,
                    "crime_types_of_interest": ["theft", "vandalism"],
                }
            ),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def get_all_features_by_group(self, feature_group: str, pattern: str = "*"):
        keys = self.client.keys(f"{feature_group}:{pattern}")
        return (
            {k.split(":")[1]: json.loads(self.client.get(k)) for k in keys}
            if keys
            else {}
        )

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))


class MockCrimePredictor:
    def __init__(self):
        pass

    def predict_risk(
        self,
        location: Dict[str, float],
        time: datetime.datetime,
        historical_data: Dict[str, Any],
    ) -> float:
        return random.uniform(0.1, 0.9)  # Mock risk score


class MockNotificationService:
    def __init__(self):
        pass

    async def send_sms_alert(self, phone_number: str, message: str):
        logging.info(f"SMS Alert to {phone_number}: {message}")

    async def send_app_notification(
        self, user_id: str, title: str, body: str, data: Dict[str, Any]
    ):
        logging.info(f"App Notification to {user_id}: {title} - {body}")

    async def dispatch_police_alert(self, incident_details: Dict[str, Any]):
        logging.critical(
            f"Police Dispatch Alert: {incident_details.get('type')} at {incident_details.get('location')}. Severity: {incident_details.get('severity_score')}"
        )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PublicSafetyAlertSystem:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.safety_config = self.config["environments"][environment][
            "public_safety_alert_system"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.crime_predictor = MockCrimePredictor()
        self.notification_service = MockNotificationService()

        self.last_incident_alerts = deque(maxlen=100)  # For recent alert tracking
        logger.info("PublicSafetyAlertSystem initialized.")

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
    public_safety_alert_system:
      enabled: true
      scan_interval_seconds: 5
      min_alert_severity: 0.7
      police_dispatch_threshold: 0.85
      city_zones:
        ZONE_X: {"lat": 35.715, "lon": 51.415, "radius_km": 1.0}
        ZONE_Y: {"lat": 35.73, "lon": 51.43, "radius_km": 1.5}
      citizen_alert_types: ["theft", "assault", "vandalism"]
      default_alert_radius_km: 0.5
      police_contact_numbers: ["+491701234567"]
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _is_within_radius(
        self, loc1: Dict[str, float], loc2: Dict[str, float], radius_km: float
    ) -> bool:
        """Calculates rough distance (km) between two lat/lon points and checks if within radius."""
        lat1, lon1 = loc1["lat"], loc1["lon"]
        lat2, lon2 = loc2["lat"], loc2["lon"]
        dist_km = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111.0  # Approx. km
        return dist_km <= radius_km

    async def _process_incident(self, incident: Dict[str, Any]):
        """Processes a detected incident and triggers alerts/dispatches."""
        incident_id = incident.get("id", f"INC_{random.randint(1000, 9999)}")
        incident_type = incident.get("type", "unknown")
        location = incident.get("location", {"lat": 0, "lon": 0})
        severity_score = incident.get("severity_score", 0.5)

        if severity_score < self.safety_config["min_alert_severity"]:
            logger.debug(
                f"Incident {incident_id} (Type: {incident_type}) below alert threshold ({severity_score:.2f})."
            )
            return

        logger.info(
            f"High-severity incident {incident_id} detected: {incident_type} at {location}. Severity: {severity_score:.2f}."
        )

        # 1. Police Dispatch (if critical)
        if severity_score >= self.safety_config["police_dispatch_threshold"]:
            await self.notification_service.dispatch_police_alert(incident)
            logger.critical(f"Police dispatched for incident {incident_id}!")

        # 2. Citizen Alerts
        if incident_type in self.safety_config["citizen_alert_types"]:
            all_citizens_prefs = self.feature_store.get_all_features_by_group(
                "citizen_preference"
            )
            for user_id, prefs in all_citizens_prefs.items():
                alert_radius = prefs.get(
                    "alert_radius_km", self.safety_config["default_alert_radius_km"]
                )
                types_of_interest = prefs.get("crime_types_of_interest", [])

                if self._is_within_radius(
                    location, prefs.get("current_location", location), alert_radius
                ) and (not types_of_interest or incident_type in types_of_interest):
                    alert_title = f"Public Safety Alert: {incident_type.replace('_', ' ').title()} Near You"
                    alert_body = f"An incident of type '{incident_type}' has been reported near {location['lat']:.4f}, {location['lon']:.4f}."
                    await self.notification_service.send_app_notification(
                        user_id, alert_title, alert_body, incident
                    )

        self.last_incident_alerts.append(
            {
                "id": incident_id,
                "timestamp": datetime.datetime.utcnow(),
                "severity": severity_score,
            }
        )
        # Update incident status in feature store
        incident["status"] = "alerted"
        self.feature_store.set_feature("incident_report", incident_id, incident)

    async def run_alert_system_loop(self):
        """Main loop for continuously monitoring for incidents and issuing alerts."""
        if not self.safety_config["enabled"]:
            logger.info("Public Safety Alert System is disabled.")
            return

        while True:
            logger.info("Scanning for public safety incidents and anomalies...")

            # 1. Check for new incident reports
            new_incident_reports = self.feature_store.get_all_features_by_group(
                "incident_report"
            )
            # Filter for incidents not yet processed by this system instance (e.g., status 'pending' or 'unprocessed')
            unprocessed_incidents = {
                id: inc
                for id, inc in new_incident_reports.items()
                if inc.get("status", "pending") == "pending"
            }

            for incident_id, incident_data in unprocessed_incidents.items():
                incident_data["id"] = incident_id  # Ensure ID is in data
                await self._process_incident(incident_data)

            # 2. Check for real-time anomaly detections (e.g., from CCTV, sensor networks)
            anomaly_detections = self.feature_store.get_all_features_by_group(
                "anomaly_detection", pattern="*:current"
            )
            for anomaly_id, anomaly_data in anomaly_detections.items():
                if (
                    anomaly_data.get("anomaly_score", 0)
                    > self.safety_config["min_alert_severity"]
                ):
                    # Create a mock incident from anomaly for processing
                    mock_incident = {
                        "id": f"ANOMALY_{anomaly_id}_{datetime.datetime.utcnow().timestamp()}",
                        "type": anomaly_data.get("type", "general_anomaly"),
                        "location": anomaly_data.get("location", {}),
                        "time": anomaly_data.get(
                            "timestamp", datetime.datetime.utcnow().isoformat() + "Z"
                        ),
                        "severity_score": anomaly_data["anomaly_score"],
                    }
                    await self._process_incident(mock_incident)
                    # For demo, mark anomaly as processed
                    anomaly_data["status"] = "processed_by_safety_system"
                    self.feature_store.set_feature(
                        "anomaly_detection", anomaly_id, anomaly_data
                    )

            await asyncio.sleep(self.safety_config["scan_interval_seconds"])


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
    public_safety_alert_system:
      enabled: true
      scan_interval_seconds: 3 # Faster for demo
      min_alert_severity: 0.6 # Lower for easier trigger
      police_dispatch_threshold: 0.8
      city_zones:
        ZONE_X: {"lat": 35.715, "lon": 51.415, "radius_km": 1.0}
        ZONE_Y: {"lat": 35.73, "lon": 51.43, "radius_km": 1.5}
      citizen_alert_types: ["theft", "assault", "vandalism", "unusual_gathering"]
      default_alert_radius_km: 1.0
      police_contact_numbers: ["+491701234567"]
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis. Populating dummy data for PublicSafetyAlertSystem.")

        # Incident that should trigger an alert
        r.set(
            "incident_report:INC_THEFT_001",
            json.dumps(
                {
                    "type": "theft",
                    "location": {"lat": 35.71, "lon": 51.41},
                    "time": datetime.datetime.utcnow().isoformat() + "Z",
                    "severity_score": 0.75,
                    "status": "pending",
                }
            ),
        )
        # Incident below threshold
        r.set(
            "incident_report:INC_MINOR_VANDALISM",
            json.dumps(
                {
                    "type": "vandalism",
                    "location": {"lat": 35.72, "lon": 51.43},
                    "time": datetime.datetime.utcnow().isoformat() + "Z",
                    "severity_score": 0.4,
                    "status": "pending",
                }
            ),
        )
        # Anomaly that's critical
        r.set(
            "anomaly_detection:CAMERA_456:current",
            json.dumps(
                {
                    "anomaly_score": 0.9,
                    "type": "violent_altercation",
                    "location": {"lat": 35.70, "lon": 51.40},
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        # Citizen preferences
        r.set(
            "citizen_preference:USER_ALICE",
            json.dumps(
                {
                    "current_location": {"lat": 35.71, "lon": 51.415},
                    "alert_radius_km": 0.5,
                    "crime_types_of_interest": ["theft", "violent_altercation"],
                }
            ),
        )
        r.set(
            "citizen_preference:USER_BOB",
            json.dumps(
                {
                    "current_location": {"lat": 35.75, "lon": 51.45},
                    "alert_radius_km": 2,
                    "crime_types_of_interest": ["vandalism"],
                }
            ),
        )

    except redis.exceptions.ConnectionError:
        print(
            "Redis not running. Public safety alert system will start with empty data."
        )

    async def main_public_safety():
        system = PublicSafetyAlertSystem(config_file)
        print("Starting PublicSafetyAlertSystem for 30 seconds...")
        try:
            await asyncio.wait_for(system.run_alert_system_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("\nPublicSafetyAlertSystem demo timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nPublicSafetyAlertSystem demo stopped by user.")

    asyncio.run(main_public_safety())
