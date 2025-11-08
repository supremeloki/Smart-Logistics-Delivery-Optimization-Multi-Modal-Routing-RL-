import datetime
import json
import yaml
import os
import logging
import random
import asyncio
import redis
from typing import Dict, Any, Tuple
import networkx as nx
import numpy as np


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "robot:RBT_001",
            json.dumps(
                {
                    "status": "idle",
                    "location_zone": "A1",
                    "battery_percent": 90,
                    "payload_capacity_kg": 20,
                    "assigned_task_id": None,
                }
            ),
        )
        self.client.set(
            "robot:RBT_002",
            json.dumps(
                {
                    "status": "charging",
                    "location_zone": "CHARGE_BAY_1",
                    "battery_percent": 30,
                    "payload_capacity_kg": 20,
                    "assigned_task_id": None,
                }
            ),
        )
        self.client.set(
            "robot:RBT_003",
            json.dumps(
                {
                    "status": "moving",
                    "location_zone": "B2",
                    "battery_percent": 75,
                    "payload_capacity_kg": 15,
                    "assigned_task_id": "TASK_PACK_001",
                }
            ),
        )
        self.client.set(
            "warehouse_task:PICK_001",
            json.dumps(
                {
                    "type": "picking",
                    "status": "pending",
                    "priority": 5,
                    "zone": "A2",
                    "item_weight_kg": 5,
                    "required_skills": ["forklift"],
                }
            ),
        )
        self.client.set(
            "warehouse_task:PACK_002",
            json.dumps(
                {
                    "type": "packing",
                    "status": "pending",
                    "priority": 8,
                    "zone": "PACK_STATION_1",
                    "item_weight_kg": 2,
                    "required_skills": ["packing_arm"],
                }
            ),
        )
        self.client.set(
            "warehouse_task:MOVE_003",
            json.dumps(
                {
                    "type": "moving",
                    "status": "pending",
                    "priority": 3,
                    "zone": "D3",
                    "item_weight_kg": 10,
                    "required_skills": ["heavy_lift"],
                }
            ),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def get_all_features_by_group(self, feature_group: str):
        keys = self.client.keys(f"{feature_group}:*")
        return (
            {k.split(":")[1]: json.loads(self.client.get(k)) for k in keys}
            if keys
            else {}
        )

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))


class MockWarehouseLayout:
    def __init__(self):
        self.zone_graph = nx.DiGraph()
        self.zone_graph.add_edges_from(
            [
                ("A1", "A2", {"travel_time_sec": 10}),
                ("A2", "PACK_STATION_1", {"travel_time_sec": 30}),
                ("A1", "CHARGE_BAY_1", {"travel_time_sec": 15}),
                ("B1", "A1", {"travel_time_sec": 20}),
                ("B2", "CHARGE_BAY_1", {"travel_time_sec": 40}),
                ("PACK_STATION_1", "SHIPPING", {"travel_time_sec": 20}),
            ]
        )

    def get_travel_time(self, start_zone: str, end_zone: str) -> float:
        try:
            return nx.shortest_path_length(
                self.zone_graph,
                source=start_zone,
                target=end_zone,
                weight="travel_time",
            )
        except nx.NetworkXNoPath:
            return np.inf
        except nx.NodeNotFound:
            return np.inf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicRoboticsOrchestrator:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.robotics_config = self.config["environments"][environment][
            "robotics_orchestrator"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.warehouse_layout = MockWarehouseLayout()

        self.pending_tasks_queue = asyncio.Queue()
        self.active_robot_tasks = {}  # robot_id -> task_id

        logger.info("DynamicRoboticsOrchestrator initialized.")

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
    robotics_orchestrator:
      enabled: true
      orchestration_interval_seconds: 5
      min_robot_battery_for_task: 20
      max_robot_payload_safety_factor: 0.8
      task_priority_weight: 1.0
      distance_cost_weight: 0.05 # Cost per second of travel time
      battery_drain_cost_weight: 0.1 # Cost per % battery drain
      skill_match_bonus: 50 # Bonus for perfect skill match
    alerting: # Dummy for mock
      slack:
        enabled: false
      email:
        enabled: false
      rules: []
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _calculate_task_cost(
        self, robot: Dict[str, Any], task: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Calculates a 'cost' (lower is better) for a robot to perform a task."""
        cost = 0.0

        # 1. Battery check
        if (
            robot["battery_percent"]
            < self.robotics_config["min_robot_battery_for_task"]
        ):
            return np.inf, "Insufficient battery"

        # 2. Payload capacity
        if (
            task.get("item_weight_kg", 0)
            > robot["payload_capacity_kg"]
            * self.robotics_config["max_robot_payload_safety_factor"]
        ):
            return np.inf, "Payload too heavy for robot"

        # 3. Skill match
        required_skills = set(task.get("required_skills", []))
        robot_skills = set(
            robot.get("skills", [])
        )  # Assume robots have a 'skills' attribute
        if not required_skills.issubset(robot_skills):
            # For creative sandbox, we'll allow partial match with penalty, or reject if strict
            if required_skills:  # If skills are required, and robot doesn't have them
                return (
                    np.inf,
                    f"Missing required skills: {', '.join(required_skills - robot_skills)}",
                )

        # 4. Travel time/distance cost
        travel_time_sec = self.warehouse_layout.get_travel_time(
            robot["location_zone"], task["zone"]
        )
        if travel_time_sec == np.inf:
            return (
                np.inf,
                f"No path from robot zone {robot['location_zone']} to task zone {task['zone']}",
            )

        cost += travel_time_sec * self.robotics_config["distance_cost_weight"]

        # 5. Battery drain cost (mock: proportional to travel time)
        # Assume 1% battery drain per 30 seconds of travel
        estimated_battery_drain_percent = travel_time_sec / 30.0
        cost += (
            estimated_battery_drain_percent
            * self.robotics_config["battery_drain_cost_weight"]
        )

        # 6. Task Priority (negative cost / bonus)
        # Higher priority tasks should have lower cost (more attractive)
        cost -= task.get("priority", 0) * self.robotics_config["task_priority_weight"]

        # 7. Skill match bonus (e.g. for perfect match or specialized skills)
        if required_skills and robot_skills.issuperset(required_skills):
            cost -= self.robotics_config[
                "skill_match_bonus"
            ]  # Reward perfect skill match

        return cost, ""

    async def _assign_task(
        self,
        robot_id: str,
        task_id: str,
        robot_data: Dict[str, Any],
        task_data: Dict[str, Any],
    ):
        """Simulates assigning a task to a robot."""
        logger.info(
            f"Assigning Task {task_id} ({task_data['type']}) to Robot {robot_id}."
        )

        # Update Feature Store: Robot status, assigned_task_id
        robot_data["status"] = "moving"  # Or 'working'
        robot_data["assigned_task_id"] = task_id
        self.feature_store.set_feature("robot", robot_id, robot_data)

        # Update Feature Store: Task status
        task_data["status"] = "assigned"
        task_data["assigned_robot_id"] = robot_id
        self.feature_store.set_feature("warehouse_task", task_id, task_data)

        # Simulate task execution time
        travel_time = self.warehouse_layout.get_travel_time(
            robot_data["location_zone"], task_data["zone"]
        )
        task_duration_sec = travel_time + random.uniform(
            30, 120
        )  # Travel + actual work time
        await asyncio.sleep(task_duration_sec)

        logger.info(f"Robot {robot_id} completed Task {task_id}.")

        # Update Feature Store: Robot status, battery, location
        robot_data["status"] = "idle"
        robot_data["assigned_task_id"] = None
        robot_data["location_zone"] = task_data["zone"]  # Robot is now at task location
        # Simulate battery drain from task
        robot_data["battery_percent"] = max(
            0, robot_data["battery_percent"] - (task_duration_sec / 60)
        )  # 1% per minute
        self.feature_store.set_feature("robot", robot_id, robot_data)

        # Update Feature Store: Task status
        task_data["status"] = "completed"
        task_data["completed_time"] = datetime.datetime.utcnow().isoformat() + "Z"
        self.feature_store.set_feature("warehouse_task", task_id, task_data)

    async def orchestration_loop(self):
        """Main loop for orchestrating warehouse robots."""
        if not self.robotics_config["enabled"]:
            logger.info("Robotics orchestrator is disabled.")
            return

        while True:
            logger.info("Orchestrating warehouse robotics tasks...")

            # 1. Fetch current state of robots and tasks
            all_robots = self.feature_store.get_all_features_by_group("robot")
            all_tasks = self.feature_store.get_all_features_by_group("warehouse_task")

            available_robots = {
                rid: r
                for rid, r in all_robots.items()
                if r.get("status") == "idle"
                and r.get("battery_percent", 0)
                >= self.robotics_config["min_robot_battery_for_task"]
            }
            pending_tasks = {
                tid: t for tid, t in all_tasks.items() if t.get("status") == "pending"
            }

            if not available_robots or not pending_tasks:
                logger.info("No available robots or pending tasks. Waiting...")
                await asyncio.sleep(
                    self.robotics_config["orchestration_interval_seconds"]
                )
                continue

            # 2. Find best assignment (robot, task) pair using cost function
            best_cost = np.inf
            best_robot_id = None
            best_task_id = None

            for robot_id, robot_data in available_robots.items():
                for task_id, task_data in pending_tasks.items():
                    cost, reason = self._calculate_task_cost(robot_data, task_data)
                    if cost < best_cost:
                        best_cost = cost
                        best_robot_id = robot_id
                        best_task_id = task_id
                        if reason:
                            logger.debug(
                                f"Robot {robot_id} cannot do task {task_id}: {reason}"
                            )  # Log if infeasible

            if best_robot_id and best_task_id and best_cost != np.inf:
                # 3. Assign the task (run as a background task)
                robot_data = available_robots[best_robot_id]
                task_data = pending_tasks[best_task_id]
                asyncio.create_task(
                    self._assign_task(
                        best_robot_id, best_task_id, robot_data, task_data
                    )
                )
                logger.info(
                    f"Assigned best pair: Robot {best_robot_id} to Task {best_task_id} with cost {best_cost:.2f}."
                )
            else:
                logger.info("No feasible robot-task assignment found in this cycle.")

            await asyncio.sleep(self.robotics_config["orchestration_interval_seconds"])


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
    robotics_orchestrator:
      enabled: true
      orchestration_interval_seconds: 5
      min_robot_battery_for_task: 20
      max_robot_payload_safety_factor: 0.8
      task_priority_weight: 1.0
      distance_cost_weight: 0.05
      battery_drain_cost_weight: 0.1
      skill_match_bonus: 50
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
        print("Connected to Redis. Populating dummy data for Robotics Orchestrator.")

        # Robots
        r.set(
            "robot:RBT_001",
            json.dumps(
                {
                    "status": "idle",
                    "location_zone": "A1",
                    "battery_percent": 90,
                    "payload_capacity_kg": 20,
                    "assigned_task_id": None,
                    "skills": ["forklift", "packing_arm"],
                }
            ),
        )
        r.set(
            "robot:RBT_002",
            json.dumps(
                {
                    "status": "idle",
                    "location_zone": "C1",
                    "battery_percent": 70,
                    "payload_capacity_kg": 10,
                    "assigned_task_id": None,
                    "skills": ["packing_arm"],
                }
            ),
        )
        r.set(
            "robot:RBT_003",
            json.dumps(
                {
                    "status": "charging",
                    "location_zone": "CHARGE_BAY_1",
                    "battery_percent": 15,
                    "payload_capacity_kg": 25,
                    "assigned_task_id": None,
                    "skills": ["heavy_lift"],
                }
            ),
        )

        # Tasks
        r.set(
            "warehouse_task:PICK_001",
            json.dumps(
                {
                    "type": "picking",
                    "status": "pending",
                    "priority": 10,
                    "zone": "A2",
                    "item_weight_kg": 5,
                    "required_skills": ["forklift"],
                }
            ),
        )
        r.set(
            "warehouse_task:PACK_002",
            json.dumps(
                {
                    "type": "packing",
                    "status": "pending",
                    "priority": 8,
                    "zone": "PACK_STATION_1",
                    "item_weight_kg": 2,
                    "required_skills": ["packing_arm"],
                }
            ),
        )
        r.set(
            "warehouse_task:MOVE_003",
            json.dumps(
                {
                    "type": "moving",
                    "status": "pending",
                    "priority": 3,
                    "zone": "D3",
                    "item_weight_kg": 18,
                    "required_skills": ["heavy_lift"],
                }
            ),
        )
        r.set(
            "warehouse_task:PICK_004_LOW_PRIO",
            json.dumps(
                {
                    "type": "picking",
                    "status": "pending",
                    "priority": 1,
                    "zone": "A3",
                    "item_weight_kg": 1,
                    "required_skills": [],
                }
            ),
        )

    except redis.exceptions.ConnectionError:
        print("Redis not running. Robotics orchestrator will start with empty data.")

    async def main_robotics_orchestrator():
        orchestrator = DynamicRoboticsOrchestrator(config_file)
        print("Starting DynamicRoboticsOrchestrator for 30 seconds...")
        try:
            await asyncio.wait_for(orchestrator.orchestration_loop(), timeout=30)
        except asyncio.TimeoutError:
            print("\nDynamicRoboticsOrchestrator demo timed out after 30 seconds.")
        except KeyboardInterrupt:
            print("\nDynamicRoboticsOrchestrator demo stopped by user.")

    asyncio.run(main_robotics_orchestrator())
