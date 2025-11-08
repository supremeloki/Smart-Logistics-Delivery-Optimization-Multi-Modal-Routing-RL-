import json
import os
import yaml
import logging
import numpy as np
import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XAIDecisionInterpreter:
    def __init__(
        self,
        config_path="conf/environments/prod.yaml",
        environment="dev",
        feature_mapping_path="conf/feature_mapping.yaml",
    ):
        self.config = self._load_config(config_path)
        self.feature_mapping = self._load_feature_mapping(feature_mapping_path)
        # Assuming we have access to the RL model's structure or can probe it
        # For this example, we'll mock 'feature_importance' or 'rule_based_explanation' logic.
        logger.info("XAIDecisionInterpreter initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    xai:
      enabled: true
      explanation_method: mock_rule_based
      explanation_level: verbose
      top_n_features: 3
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_feature_mapping(self, feature_mapping_path):
        # This file would define how raw observation indices map to human-readable features
        if not os.path.exists(feature_mapping_path):
            os.makedirs(os.path.dirname(feature_mapping_path), exist_ok=True)
            with open(feature_mapping_path, "w") as f:
                f.write(
                    """
observation_features:
  driver_policy:
    0: "current_node_id"
    1: "current_load_kg"
    2: "driver_status_available"
    3: "driver_speed_mps"
    4: "order_1_origin_node"
    5: "order_1_destination_node"
    6: "order_1_weight"
    7: "order_1_pickup_deadline_relative"
    8: "order_1_delivery_deadline_relative"
    9: "traffic_factor_avg_near_driver"
    10: "demand_density_near_driver"
"""
                )
        with open(feature_mapping_path, "r") as f:
            return yaml.safe_load(f)

    def _mock_feature_importance(
        self, obs: np.ndarray, action: str, policy_id: str
    ) -> Dict[str, float]:
        """
        Simulates feature importance for an action. In a real system, this would use SHAP/LIME.
        """
        feature_names = self.feature_mapping.get("observation_features", {}).get(
            policy_id, {}
        )
        importance = {}

        # Assign random importance for demo, with some bias towards relevant features
        for i in range(len(obs)):
            feat_name = feature_names.get(i, f"feature_{i}")
            if "node" in feat_name or "demand" in feat_name:
                importance[feat_name] = np.random.uniform(0.5, 1.0) * np.random.choice(
                    [-1, 1]
                )
            elif "load" in feat_name or "speed" in feat_name:
                importance[feat_name] = np.random.uniform(0.3, 0.8) * np.random.choice(
                    [-1, 1]
                )
            else:
                importance[feat_name] = np.random.uniform(0.1, 0.5) * np.random.choice(
                    [-1, 1]
                )

        # Introduce bias based on action for creativity
        if "move_to_node" in action:
            importance["traffic_factor_avg_near_driver"] = abs(
                importance.get("traffic_factor_avg_near_driver", 0.8)
            ) * (
                1 if obs[9] < 1.2 else -1
            )  # Favorable traffic
            importance["demand_density_near_driver"] = abs(
                importance.get("demand_density_near_driver", 0.7)
            )  # High demand
        elif "assign_order" in action:
            importance["order_1_pickup_deadline_relative"] = abs(
                importance.get("order_1_pickup_deadline_relative", 0.9)
            ) * (
                -1 if obs[7] < 30 else 1
            )  # Urgent pickup
            importance["driver_status_available"] = abs(
                importance.get("driver_status_available", 0.9)
            )  # Driver is available
        elif "wait" in action:
            importance["demand_density_near_driver"] = -abs(
                importance.get("demand_density_near_driver", 0.7)
            )  # Low demand
            importance["driver_status_available"] = abs(
                importance.get("driver_status_available", 0.5)
            )  # But available

        return importance

    def _generate_rule_based_explanation(
        self, obs: np.ndarray, action: str, policy_id: str
    ) -> str:
        """
        Generates a human-readable explanation based on predefined rules or heuristic.
        """
        feature_names = self.feature_mapping.get("observation_features", {}).get(
            policy_id, {}
        )

        explanation = [f"The agent chose to '{action}' because:"]

        # Example rule for 'move_to_node'
        if (
            "move_to_node" in action
            and "traffic_factor_avg_near_driver" in feature_names.values()
            and "demand_density_near_driver" in feature_names.values()
        ):
            traffic_idx = list(feature_names.keys())[
                list(feature_names.values()).index("traffic_factor_avg_near_driver")
            ]
            demand_idx = list(feature_names.keys())[
                list(feature_names.values()).index("demand_density_near_driver")
            ]

            if obs[traffic_idx] < 1.0:
                explanation.append(
                    f"- Traffic conditions towards the target node ({action.split('_')[-1]}) are currently favorable (factor: {obs[traffic_idx]:.2f})."
                )
            if obs[demand_idx] > 0.5:  # Assuming >0.5 indicates high demand
                explanation.append(
                    f"- There is high order demand in the vicinity of the target node (density: {obs[demand_idx]:.2f})."
                )
            if "driver_load_kg" in feature_names.values():
                load_idx = list(feature_names.keys())[
                    list(feature_names.values()).index("current_load_kg")
                ]
                if obs[load_idx] < 10:
                    explanation.append(
                        f"- The driver has sufficient capacity to take on new orders (current load: {obs[load_idx]:.1f} kg)."
                    )

        # Example rule for 'assign_order'
        elif (
            "assign_order" in action
            and "order_1_pickup_deadline_relative" in feature_names.values()
        ):
            pickup_deadline_idx = list(feature_names.keys())[
                list(feature_names.values()).index("order_1_pickup_deadline_relative")
            ]
            if (
                obs[pickup_deadline_idx] < 30
            ):  # If pickup deadline is less than 30 relative units (e.g., minutes)
                explanation.append(
                    f"- The assigned order ({action.split('_')[-1]}) has an urgent pickup deadline (remaining: {obs[pickup_deadline_idx]:.0f} min)."
                )
            if "driver_status_available" in feature_names.values():
                status_idx = list(feature_names.keys())[
                    list(feature_names.values()).index("driver_status_available")
                ]
                if obs[status_idx] == 1:
                    explanation.append(
                        f"- An available driver was identified for this assignment."
                    )

        # Example rule for 'wait'
        elif (
            action == "wait" and "demand_density_near_driver" in feature_names.values()
        ):
            demand_idx = list(feature_names.keys())[
                list(feature_names.values()).index("demand_density_near_driver")
            ]
            if obs[demand_idx] < 0.2:
                explanation.append(
                    f"- There is currently low demand for new orders in the driver's immediate area (density: {obs[demand_idx]:.2f})."
                )
            explanation.append(
                "- The driver is optimally positioned to await new opportunities."
            )

        if len(explanation) == 1:  # Only the initial phrase exists
            explanation.append(
                "- No specific prominent factors identified for this decision based on current rules."
            )

        return "\n".join(explanation)

    def interpret_decision(
        self, obs: np.ndarray, action: str, policy_id: str = "driver_policy"
    ) -> Dict[str, Any]:
        """
        Provides an explanation for a given RL agent's decision.
        """
        xai_config = self.config["environments"]["dev"]["xai"]
        method = xai_config.get("explanation_method", "mock_rule_based")
        xai_config.get("explanation_level", "verbose")
        top_n = xai_config.get("top_n_features", 3)

        explanation_text = ""
        feature_importance_scores = {}

        if method == "mock_feature_importance":
            feature_importance_scores = self._mock_feature_importance(
                obs, action, policy_id
            )
            sorted_importance = sorted(
                feature_importance_scores.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )

            explanation_parts = [f"The agent chose '{action}' because:"]
            for feat, score in sorted_importance[:top_n]:
                direction = (
                    "positively influenced" if score > 0 else "negatively influenced"
                )
                explanation_parts.append(
                    f"- Feature '{feat}' {direction} the decision (score: {score:.2f})."
                )
            explanation_text = "\n".join(explanation_parts)

        elif method == "mock_rule_based":
            explanation_text = self._generate_rule_based_explanation(
                obs, action, policy_id
            )
            # You could also integrate feature importance here if desired, but keep it simple for rule-based.

        return {
            "action": action,
            "policy_id": policy_id,
            "explanation_method": method,
            "explanation_text": explanation_text,
            "feature_importance_scores": (
                feature_importance_scores
                if method == "mock_feature_importance"
                else None
            ),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    feature_map_file = "conf/feature_mapping.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(feature_map_file), exist_ok=True)

    # Create dummy config and feature mapping if they don't exist
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    xai:
      enabled: true
      explanation_method: mock_rule_based
      explanation_level: verbose
      top_n_features: 3
"""
            )
    if not os.path.exists(feature_map_file):
        with open(feature_map_file, "w") as f:
            f.write(
                """
observation_features:
  driver_policy:
    0: "current_node_id"
    1: "current_load_kg"
    2: "driver_status_available"
    3: "driver_speed_mps"
    4: "order_1_origin_node"
    5: "order_1_destination_node"
    6: "order_1_weight"
    7: "order_1_pickup_deadline_relative"
    8: "order_1_delivery_deadline_relative"
    9: "traffic_factor_avg_near_driver"
    10: "demand_density_near_driver"
"""
            )

    interpreter = XAIDecisionInterpreter(
        config_file, feature_mapping_path=feature_map_file
    )

    # Simulate an observation and an action
    # This observation array should match the feature_mapping's indices
    mock_obs = np.array(
        [
            100,  # current_node_id (dummy value)
            5.2,  # current_load_kg
            1,  # driver_status_available (1=available)
            12.5,  # driver_speed_mps
            100,  # order_1_origin_node
            150,  # order_1_destination_node
            2.1,  # order_1_weight
            20,  # order_1_pickup_deadline_relative (minutes)
            120,  # order_1_delivery_deadline_relative (minutes)
            0.8,  # traffic_factor_avg_near_driver (0.8 = good traffic)
            0.7,  # demand_density_near_driver (0.7 = high demand)
        ],
        dtype=np.float32,
    )

    # Test with a "move_to_node" action
    action_move = "move_to_node_200"
    explanation_move = interpreter.interpret_decision(mock_obs, action_move)
    print("--- Explanation for 'move_to_node' ---")
    print(json.dumps(explanation_move, indent=2, ensure_ascii=False))

    # Test with an "assign_order" action
    mock_obs_assign = np.array(
        [
            100,
            5.2,
            1,
            12.5,
            100,
            150,
            2.1,
            10,
            120,
            1.1,
            0.3,  # Urgent pickup, bad traffic, low demand
        ],
        dtype=np.float32,
    )
    action_assign = "assign_order_ORD456"
    explanation_assign = interpreter.interpret_decision(mock_obs_assign, action_assign)
    print("\n--- Explanation for 'assign_order' ---")
    print(json.dumps(explanation_assign, indent=2, ensure_ascii=False))

    # Test with a "wait" action
    mock_obs_wait = np.array(
        [
            100,
            5.2,
            1,
            12.5,
            100,
            150,
            2.1,
            60,
            180,
            1.0,
            0.1,  # Not urgent, normal traffic, very low demand
        ],
        dtype=np.float32,
    )
    action_wait = "wait"
    explanation_wait = interpreter.interpret_decision(mock_obs_wait, action_wait)
    print("\n--- Explanation for 'wait' ---")
    print(json.dumps(explanation_wait, indent=2, ensure_ascii=False))

    # Switch to mock_feature_importance method
    with open(config_file, "w") as f:
        f.write(
            """
environments:
  dev:
    xai:
      enabled: true
      explanation_method: mock_feature_importance
      explanation_level: verbose
      top_n_features: 3
"""
        )
    interpreter_fi = XAIDecisionInterpreter(
        config_file, feature_mapping_path=feature_map_file
    )
    explanation_fi = interpreter_fi.interpret_decision(mock_obs, action_move)
    print("\n--- Explanation using 'mock_feature_importance' for 'move_to_node' ---")
    print(json.dumps(explanation_fi, indent=2, ensure_ascii=False))
