import jsonschema
import datetime
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestionValidator:
    def __init__(self):
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        # In a real system, these would be loaded from files (e.g., JSON Schema files)
        # For demonstration, define them inline.

        # Traffic Event Schema
        traffic_event_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "format": "date-time"},
                "u": {"type": "integer", "minimum": 0},
                "v": {"type": "integer", "minimum": 0},
                "key": {"type": "integer", "minimum": 0},
                "current_travel_time": {"type": "number", "minimum": 0},
            },
            "required": ["timestamp", "u", "v", "key", "current_travel_time"],
            "additionalProperties": False,
        }

        # Order Event Schema
        order_event_schema = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "origin_node": {"type": "integer", "minimum": 0},
                "destination_node": {"type": "integer", "minimum": 0},
                "weight": {"type": "number", "minimum": 0},
                "volume": {"type": "number", "minimum": 0},
                "pickup_time_start": {"type": "string", "format": "date-time"},
                "pickup_time_end": {"type": "string", "format": "date-time"},
                "delivery_time_latest": {"type": "string", "format": "date-time"},
                "status": {
                    "type": "string",
                    "enum": [
                        "pending",
                        "assigned",
                        "picked_up",
                        "delivered",
                        "cancelled",
                    ],
                },
            },
            "required": [
                "order_id",
                "origin_node",
                "destination_node",
                "weight",
                "volume",
                "pickup_time_start",
                "pickup_time_end",
                "delivery_time_latest",
                "status",
            ],
            "additionalProperties": False,
        }

        # Vehicle Telemetry Schema
        telemetry_event_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "format": "date-time"},
                "driver_id": {"type": "string"},
                "current_node_id": {"type": "integer", "minimum": 0},
                "speed_kph": {"type": "number", "minimum": 0},
                "fuel_level_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "load_kg": {"type": "number", "minimum": 0},
            },
            "required": [
                "timestamp",
                "driver_id",
                "current_node_id",
                "speed_kph",
                "fuel_level_percent",
                "load_kg",
            ],
            "additionalProperties": False,
        }

        return {
            "traffic_event": traffic_event_schema,
            "order_event": order_event_schema,
            "telemetry_event": telemetry_event_schema,
        }

    def validate(self, data: Dict[str, Any], schema_name: str) -> bool:
        schema = self.schemas.get(schema_name)
        if not schema:
            logger.error(f"Schema '{schema_name}' not found for validation.")
            return False

        try:
            jsonschema.validate(instance=data, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(
                f"Validation Error for {schema_name}: {e.message} in {e.path}"
            )
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during validation for {schema_name}: {e}"
            )
            return False

    def validate_and_parse_datetime(
        self, datetime_string: str
    ) -> Optional[datetime.datetime]:
        try:
            # ISO 8601 format including 'Z' for UTC or timezone offset
            return datetime.datetime.fromisoformat(
                datetime_string.replace("Z", "+00:00")
            )
        except ValueError:
            logger.warning(f"Invalid ISO 8601 datetime format: {datetime_string}")
            return None


if __name__ == "__main__":
    validator = DataIngestionValidator()

    print("--- Testing Traffic Event Validation ---")
    valid_traffic_event = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "u": 1,
        "v": 2,
        "key": 0,
        "current_travel_time": 120.5,
    }
    invalid_traffic_event_missing_key = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "u": 1,
        "v": 2,
        "current_travel_time": 120.5,
    }
    invalid_traffic_event_bad_type = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "u": "node1",
        "v": 2,
        "key": 0,
        "current_travel_time": 120.5,
    }
    invalid_traffic_event_negative_time = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "u": 1,
        "v": 2,
        "key": 0,
        "current_travel_time": -10,
    }

    print(
        f"Valid traffic event: {validator.validate(valid_traffic_event, 'traffic_event')}"
    )
    print(
        f"Invalid traffic event (missing key): {validator.validate(invalid_traffic_event_missing_key, 'traffic_event')}"
    )
    print(
        f"Invalid traffic event (bad type): {validator.validate(invalid_traffic_event_bad_type, 'traffic_event')}"
    )
    print(
        f"Invalid traffic event (negative time): {validator.validate(invalid_traffic_event_negative_time, 'traffic_event')}"
    )

    print("\n--- Testing Order Event Validation ---")
    valid_order_event = {
        "order_id": "ORD123",
        "origin_node": 10,
        "destination_node": 20,
        "weight": 5.0,
        "volume": 0.1,
        "pickup_time_start": datetime.datetime.utcnow().isoformat() + "Z",
        "pickup_time_end": (
            datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        ).isoformat()
        + "Z",
        "delivery_time_latest": (
            datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        ).isoformat()
        + "Z",
        "status": "pending",
    }
    invalid_order_event_bad_status = {
        "order_id": "ORD123",
        "origin_node": 10,
        "destination_node": 20,
        "weight": 5.0,
        "volume": 0.1,
        "pickup_time_start": datetime.datetime.utcnow().isoformat() + "Z",
        "pickup_time_end": (
            datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        ).isoformat()
        + "Z",
        "delivery_time_latest": (
            datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        ).isoformat()
        + "Z",
        "status": "in_progress",  # Not in enum
    }
    print(f"Valid order event: {validator.validate(valid_order_event, 'order_event')}")
    print(
        f"Invalid order event (bad status): {validator.validate(invalid_order_event_bad_status, 'order_event')}"
    )

    print("\n--- Testing Datetime Parsing ---")
    dt_str_utc = datetime.datetime.utcnow().isoformat() + "Z"
    dt_obj = validator.validate_and_parse_datetime(dt_str_utc)
    print(f"Parsed UTC datetime '{dt_str_utc}': {dt_obj}")

    dt_str_local = "2025-10-21T14:30:00+03:30"
    dt_obj_local = validator.validate_and_parse_datetime(dt_str_local)
    print(f"Parsed local datetime '{dt_str_local}': {dt_obj_local}")

    invalid_dt_str = "not-a-date"
    invalid_dt_obj = validator.validate_and_parse_datetime(invalid_dt_str)
    print(f"Parsed invalid datetime '{invalid_dt_str}': {invalid_dt_obj}")
