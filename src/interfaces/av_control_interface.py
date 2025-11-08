import datetime
import json
import yaml
import os
import logging
import asyncio
import random
from typing import Dict, Any
import redis


# Mock components
class MockAVCommunicationAPI:
    def __init__(self, av_id: str):
        self.av_id = av_id
        logging.info(f"MockAVCommunicationAPI for {av_id} initialized.")

    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates sending a command to an AV and getting an acknowledgment."""
        if command.get("type") == "navigate_to_coords":
            target_lat = command["params"]["target_lat"]
            target_lon = command["params"]["target_lon"]
            logging.info(
                f"[{self.av_id}] Simulating navigation to ({target_lat:.4f}, {target_lon:.4f})."
            )
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate network delay
            return {
                "status": "ACK",
                "command_id": command.get("command_id"),
                "message": "Navigation initiated.",
            }
        elif command.get("type") == "set_speed_limit":
            new_limit = command["params"]["speed_kph"]
            logging.info(
                f"[{self.av_id}] Simulating setting speed limit to {new_limit} kph."
            )
            await asyncio.sleep(random.uniform(0.05, 0.2))
            return {
                "status": "ACK",
                "command_id": command.get("command_id"),
                "message": "Speed limit applied.",
            }
        elif command.get("type") == "get_status":
            return self._generate_mock_status()
        else:
            logging.warning(
                f"[{self.av_id}] Unknown command type: {command.get('type')}"
            )
            return {
                "status": "NACK",
                "command_id": command.get("command_id"),
                "message": "Unknown command.",
            }

    async def get_telemetry_stream(self) -> Dict[str, Any]:
        """Simulates receiving a real-time telemetry update from an AV."""
        await asyncio.sleep(random.uniform(0.5, 1.5))  # Simulate telemetry interval
        return self._generate_mock_status()

    def _generate_mock_status(self) -> Dict[str, Any]:
        """Generates random mock status data for the AV."""
        lat, lon = random.uniform(35.5, 35.9), random.uniform(51.2, 51.7)
        speed = random.uniform(0, 80)
        battery = random.randint(10, 100)
        fuel = random.randint(0, 100)
        return {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "av_id": self.av_id,
            "current_location": {"lat": lat, "lon": lon},
            "speed_kph": round(speed, 1),
            "heading_degrees": round(random.uniform(0, 360), 1),
            "battery_percent": battery,
            "fuel_percent": fuel,
            "payload_status": {
                "weight_kg": random.uniform(0, 500),
                "items_count": random.randint(0, 5),
            },
            "system_health": random.choice(["optimal", "minor_warning"]),
            "is_manual_override": random.random()
            < 0.01,  # 1% chance of manual override
        }


class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "av:AV_001",
            json.dumps(
                {
                    "status": "idle",
                    "current_location": {"lat": 35.7, "lon": 51.4},
                    "battery_percent": 90,
                }
            ),
        )

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AVControlInterface:
    def __init__(
        self, av_id: str, config_path="conf/environments/prod.yaml", environment="dev"
    ):
        self.av_id = av_id
        self.config = self._load_config(config_path)
        self.av_config = self.config["environments"][environment][
            "av_control_interface"
        ]

        self.av_comm_api = MockAVCommunicationAPI(av_id)
        self.feature_store = MockFeatureStoreClient(config_path, environment)

        self.last_command_id = 0
        self.pending_commands = {}  # command_id -> command_details

        logger.info(f"AVControlInterface for AV {self.av_id} initialized.")

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
    av_control_interface:
      enabled: true
      telemetry_sync_interval_seconds: 2
      command_timeout_seconds: 10
      max_retry_attempts: 3
      critical_health_threshold: 0.8 # Score, below this means critical
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _generate_command_id(self) -> str:
        self.last_command_id += 1
        return f"CMD_{self.av_id}_{self.last_command_id}"

    async def send_navigation_command(
        self, target_lat: float, target_lon: float, mission_id: str = None
    ) -> Dict[str, Any]:
        """Sends a command to the AV to navigate to a specific coordinate."""
        if not self.av_config["enabled"]:
            logger.info(
                f"AV control disabled. Not sending navigation command to {self.av_id}."
            )
            return {"status": "DISABLED", "message": "AV control is disabled."}

        command_id = self._generate_command_id()
        command = {
            "command_id": command_id,
            "type": "navigate_to_coords",
            "params": {"target_lat": target_lat, "target_lon": target_lon},
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "mission_id": mission_id,
        }

        self.pending_commands[command_id] = {
            "command": command,
            "retries": 0,
            "sent_time": datetime.datetime.utcnow(),
        }
        logger.info(
            f"[{self.av_id}] Sending navigation command {command_id} to ({target_lat:.4f}, {target_lon:.4f})."
        )

        # Simulate sending and awaiting response (with retries)
        for attempt in range(self.av_config["max_retry_attempts"]):
            try:
                response = await asyncio.wait_for(
                    self.av_comm_api.send_command(command),
                    timeout=self.av_config["command_timeout_seconds"],
                )
                if response.get("status") == "ACK":
                    del self.pending_commands[command_id]
                    logger.info(
                        f"[{self.av_id}] Command {command_id} acknowledged: {response.get('message')}"
                    )
                    return response
                else:
                    logger.warning(
                        f"[{self.av_id}] Command {command_id} NACK: {response.get('message')}. Retrying..."
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[{self.av_id}] Command {command_id} timed out. Retrying..."
                )
            except Exception as e:
                logger.error(
                    f"[{self.av_id}] Error sending command {command_id}: {e}. Retrying..."
                )

            self.pending_commands[command_id]["retries"] += 1
            await asyncio.sleep(1 + attempt * 0.5)  # Backoff

        logger.error(
            f"[{self.av_id}] Command {command_id} failed after {self.av_config['max_retry_attempts']} attempts."
        )
        del self.pending_commands[command_id]
        return {
            "status": "FAILED",
            "command_id": command_id,
            "message": "Command failed after multiple retries.",
        }

    async def _process_telemetry_stream(self):
        """Continuously receives and processes telemetry from the AV."""
        while True:
            try:
                telemetry = await self.av_comm_api.get_telemetry_stream()
                if telemetry:
                    # Store telemetry in feature store
                    self.feature_store.set_feature(
                        "av_telemetry", self.av_id, telemetry
                    )
                    self.feature_store.set_feature(
                        "av",
                        self.av_id,
                        {
                            "status": "operating",  # Placeholder
                            "current_location": telemetry["current_location"],
                            "speed_kph": telemetry["speed_kph"],
                            "battery_percent": telemetry["battery_percent"],
                            "system_health": telemetry["system_health"],
                            "timestamp": telemetry["timestamp"],
                        },
                    )

                    # Basic health monitoring
                    if telemetry["system_health"] == "minor_warning":
                        logger.warning(
                            f"[{self.av_id}] AV system health warning: {telemetry['system_health']}. Battery: {telemetry['battery_percent']:.1f}%"
                        )
                    elif telemetry["is_manual_override"]:
                        logger.critical(
                            f"[{self.av_id}] CRITICAL ALERT: Manual override detected for AV {self.av_id}!"
                        )

            except Exception as e:
                logger.error(
                    f"Error processing telemetry for AV {self.av_id}: {e}",
                    exc_info=True,
                )

            await asyncio.sleep(self.av_config["telemetry_sync_interval_seconds"])

    async def run_interface(self):
        """Starts the AV control interface, including telemetry processing."""
        if not self.av_config["enabled"]:
            logger.info(f"AV control interface for {self.av_id} is disabled.")
            return

        telemetry_task = asyncio.create_task(self._process_telemetry_stream())

        try:
            # Keep the main interface running to allow sending commands
            await telemetry_task  # In a real system, this would be managed by an orchestrator
            # that calls send_navigation_command on demand. For this demo, we let it run.
        except asyncio.CancelledError:
            logger.info(f"AVControlInterface for {self.av_id} cancelled.")
        except Exception as e:
            logger.error(
                f"AVControlInterface run error for {self.av_id}: {e}", exc_info=True
            )


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
    av_control_interface:
      enabled: true
      telemetry_sync_interval_seconds: 1 # Faster for demo
      command_timeout_seconds: 5
      max_retry_attempts: 2
      critical_health_threshold: 0.8
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis. Populating dummy data for AVControlInterface.")
        r.set(
            "av:AV_001",
            json.dumps(
                {
                    "status": "idle",
                    "current_location": {"lat": 35.7, "lon": 51.4},
                    "battery_percent": 90,
                }
            ),
        )
    except redis.exceptions.ConnectionError:
        print("Redis not running. AVControlInterface will start with empty data.")

    async def main_av_control():
        av_id = "AV_001"
        av_interface = AVControlInterface(av_id, config_file)
        print(f"Starting AVControlInterface for {av_id}...")

        telemetry_monitor_task = asyncio.create_task(
            av_interface._process_telemetry_stream()
        )

        # Simulate sending a navigation command after a few seconds
        await asyncio.sleep(3)
        print(f"\n--- Sending Navigation Command to {av_id} ---")
        nav_response = await av_interface.send_navigation_command(
            35.75, 51.45, mission_id="MISSION_007"
        )
        print(f"Navigation Command Response: {nav_response}")

        await asyncio.sleep(5)
        print(f"\n--- Sending Another Command (Speed Limit) ---")
        speed_cmd_response = await av_interface.av_comm_api.send_command(
            {
                "command_id": av_interface._generate_command_id(),
                "type": "set_speed_limit",
                "params": {"speed_kph": 50},
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        print(f"Speed Limit Command Response: {speed_cmd_response}")

        print(
            f"\nAVControlInterface demo for {av_id} will run telemetry for 15 more seconds..."
        )
        try:
            await telemetry_monitor_task
        except asyncio.TimeoutError:
            print(f"\nAVControlInterface telemetry monitor timed out.")
        except KeyboardInterrupt:
            print(f"\nAVControlInterface demo stopped by user.")

    asyncio.run(main_av_control())
