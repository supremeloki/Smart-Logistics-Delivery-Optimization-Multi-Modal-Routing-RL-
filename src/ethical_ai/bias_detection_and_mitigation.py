import datetime
import yaml
import os
import logging
import random
from typing import Dict, Any
from collections import deque
import pandas as pd
import json


# Mocking a hypothetical RL Agent's decision log
class MockRLDecisionLog:
    def __init__(self, log_size=1000):
        self.log = deque(maxlen=log_size)
        self.driver_ids = [f"driver_{i}" for i in range(5)]
        self.node_ids = [f"node_{i}" for i in range(10)]
        self.order_ids = [f"order_{i}" for i in range(20)]

    def add_decision(self, decision: Dict[str, Any]):
        self.log.append(decision)

    def generate_dummy_decisions(self, num_decisions=100):
        for _ in range(num_decisions):
            driver_id = random.choice(self.driver_ids)
            current_node = random.choice(self.node_ids)

            # Simulate features that could lead to bias
            is_urban_area = random.random() < 0.7  # Simulate some areas being "urban"
            driver_experience = random.randint(1, 10)  # Years
            vehicle_age = random.randint(1, 15)  # Years

            action_type = random.choice(
                ["assign_high_payout", "assign_low_payout", "move_to_demand", "wait"]
            )
            payout = random.uniform(10, 100)
            if action_type == "assign_high_payout":
                payout = random.uniform(80, 200)

            decision = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "driver_id": driver_id,
                "current_location_node": current_node,
                "driver_experience_years": driver_experience,
                "vehicle_age_years": vehicle_age,
                "is_urban_area": is_urban_area,
                "chosen_action": action_type,
                "payout_offered": payout,
                "estimated_travel_time_minutes": random.uniform(10, 60),
                "predicted_traffic_factor": random.uniform(0.8, 1.5),
                "region_income_level": (
                    random.choice(["low", "medium", "high"])
                    if is_urban_area
                    else "rural"
                ),  # Simulate demographic
                "demand_level_at_node": random.uniform(0.1, 1.0),
            }
            self.add_decision(decision)
        return list(self.log)  # Return current log as list for easier processing


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasDetectionAndMitigation:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.bias_config = self.config["environments"][environment]["ethical_ai"][
            "bias_management"
        ]

        self.decision_log = MockRLDecisionLog(
            log_size=self.bias_config["history_window_size"]
        )  # Using mock log

        self.bias_metrics_history = deque(maxlen=100)  # Store historical bias scores
        logger.info("BiasDetectionAndMitigation initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    ethical_ai:
      bias_management:
        enabled: true
        history_window_size: 1000
        fairness_thresholds:
          payout_disparity_ratio: 0.8 # Min ratio of avg payout for disadvantaged vs advantaged group
          high_payout_allocation_disparity: 0.2 # Max diff in % high payout allocation
          stress_exposure_disparity: 0.1 # Max diff in % drivers above stress threshold
        mitigation_strategy: "re_weight_costs" # Example: re_weight_costs, constrained_optimization
        protected_attributes:
          - driver_experience_years
          - region_income_level
          - vehicle_age_years
        disadvantaged_groups:
          driver_experience_years: {"label": "junior_drivers", "condition": "<5"}
          region_income_level: {"label": "low_income_areas", "condition": "low"}
          vehicle_age_years: {"label": "older_vehicles", "condition": ">10"}
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _identify_protected_groups(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Identifies individuals belonging to protected/disadvantaged groups."""
        groups = {}
        for attr, details in self.bias_config["disadvantaged_groups"].items():
            if attr not in df.columns:
                logger.warning(
                    f"Protected attribute '{attr}' not found in decision log."
                )
                continue

            condition_str = details["condition"]

            # Simple parsing of conditions for demo
            if "<" in condition_str:
                threshold = float(condition_str.split("<")[1])
                groups[details["label"]] = df[attr] < threshold
            elif ">" in condition_str:
                threshold = float(condition_str.split(">")[1])
                groups[details["label"]] = df[attr] > threshold
            elif condition_str in df[attr].unique():  # Exact match for categorical
                groups[details["label"]] = df[attr] == condition_str
            else:
                logger.warning(
                    f"Unsupported condition '{condition_str}' for attribute '{attr}'."
                )
                groups[details["label"]] = pd.Series(
                    [False] * len(df)
                )  # No one matches
        return groups

    def _calculate_payout_disparity(
        self, df: pd.DataFrame, disadvantaged_group: pd.Series
    ) -> float:
        """Calculates the ratio of average payout for disadvantaged vs. advantaged group."""
        advantaged_group = ~disadvantaged_group

        avg_payout_disadvantaged = df[disadvantaged_group]["payout_offered"].mean()
        avg_payout_advantaged = df[advantaged_group]["payout_offered"].mean()

        if (
            pd.isna(avg_payout_disadvantaged)
            or pd.isna(avg_payout_advantaged)
            or avg_payout_advantaged == 0
        ):
            return 1.0  # No disparity or insufficient data

        return avg_payout_disadvantaged / avg_payout_advantaged

    def _calculate_high_payout_allocation_disparity(
        self,
        df: pd.DataFrame,
        disadvantaged_group: pd.Series,
        high_payout_threshold: float = 100.0,
    ) -> float:
        """Calculates the absolute difference in percentage of high payout orders allocated."""

        total_high_payout = (df["payout_offered"] > high_payout_threshold).sum()
        if total_high_payout == 0:
            return 0.0
        advantaged_group = ~disadvantaged_group

        disadvantaged_high_payout_count = (
            (df["payout_offered"] > high_payout_threshold) & disadvantaged_group
        ).sum()
        advantaged_high_payout_count = (
            (df["payout_offered"] > high_payout_threshold) & advantaged_group
        ).sum()

        disadvantaged_total_count = disadvantaged_group.sum()
        advantaged_total_count = advantaged_group.sum()

        if disadvantaged_total_count == 0 or advantaged_total_count == 0:
            return 0.0

        percent_disadvantaged_high_payout = (
            disadvantaged_high_payout_count / disadvantaged_total_count
        )
        percent_advantaged_high_payout = (
            advantaged_high_payout_count / advantaged_total_count
        )

        return abs(percent_disadvantaged_high_payout - percent_advantaged_high_payout)

    def _calculate_stress_exposure_disparity(
        self,
        df: pd.DataFrame,
        disadvantaged_group: pd.Series,
        stress_proxy_threshold: float = 0.8,
    ) -> float:
        """Calculates disparity in exposing groups to high stress (e.g., long travel times, high traffic)."""

        # Proxy for stress: long travel time + high traffic factor
        is_high_stress_route = (df["estimated_travel_time_minutes"] > 45) & (
            df["predicted_traffic_factor"] > 1.2
        )
        advantaged_group = ~disadvantaged_group

        disadvantaged_stress_count = (is_high_stress_route & disadvantaged_group).sum()
        advantaged_stress_count = (is_high_stress_route & advantaged_group).sum()

        disadvantaged_total_count = disadvantaged_group.sum()
        advantaged_total_count = advantaged_group.sum()

        if disadvantaged_total_count == 0 or advantaged_total_count == 0:
            return 0.0

        percent_disadvantaged_stress = (
            disadvantaged_stress_count / disadvantaged_total_count
        )
        percent_advantaged_stress = advantaged_stress_count / advantaged_total_count

        return abs(percent_disadvantaged_stress - percent_advantaged_stress)

    def detect_bias(self, decisions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects various forms of bias in the RL agent's decision-making.
        """
        if decisions_df.empty:
            logger.info("No decisions to analyze for bias.")
            return {"bias_detected": False, "details": "No decision data."}

        bias_scores = {}
        bias_alerts = []

        # Assuming 'payout_offered' column exists for financial fairness
        # Assuming 'estimated_travel_time_minutes' and 'predicted_traffic_factor' for stress

        for attr, details in self.bias_config["disadvantaged_groups"].items():
            if attr not in decisions_df.columns:
                continue

            disadvantaged_group_mask = self._identify_protected_groups(decisions_df)[
                details["label"]
            ]
            if (
                disadvantaged_group_mask.sum() == 0
                or (~disadvantaged_group_mask).sum() == 0
            ):
                logger.debug(
                    f"Insufficient data for {details['label']} group in {attr} to check for bias."
                )
                continue

            # Payout disparity
            payout_ratio = self._calculate_payout_disparity(
                decisions_df, disadvantaged_group_mask
            )
            bias_scores[f"{details['label']}_payout_disparity_ratio"] = payout_ratio
            if (
                payout_ratio
                < self.bias_config["fairness_thresholds"]["payout_disparity_ratio"]
            ):
                bias_alerts.append(
                    f"Financial bias detected for {details['label']}: Avg payout ratio {payout_ratio:.2f} (below threshold {self.bias_config['fairness_thresholds']['payout_disparity_ratio']})."
                )

            # High payout allocation disparity
            high_payout_disparity = self._calculate_high_payout_allocation_disparity(
                decisions_df, disadvantaged_group_mask
            )
            bias_scores[f"{details['label']}_high_payout_allocation_disparity"] = (
                high_payout_disparity
            )
            if (
                high_payout_disparity
                > self.bias_config["fairness_thresholds"][
                    "high_payout_allocation_disparity"
                ]
            ):
                bias_alerts.append(
                    f"High payout allocation bias for {details['label']}: Difference in % high payout {high_payout_disparity:.2f} (above threshold {self.bias_config['fairness_thresholds']['high_payout_allocation_disparity']})."
                )

            # Stress exposure disparity
            stress_disparity = self._calculate_stress_exposure_disparity(
                decisions_df, disadvantaged_group_mask
            )
            bias_scores[f"{details['label']}_stress_exposure_disparity"] = (
                stress_disparity
            )
            if (
                stress_disparity
                > self.bias_config["fairness_thresholds"]["stress_exposure_disparity"]
            ):
                bias_alerts.append(
                    f"Stress exposure bias for {details['label']}: Difference in % high stress routes {stress_disparity:.2f} (above threshold {self.bias_config['fairness_thresholds']['stress_exposure_disparity']})."
                )

        is_any_bias_detected = len(bias_alerts) > 0

        result = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "bias_detected": is_any_bias_detected,
            "bias_scores": bias_scores,
            "bias_alerts": bias_alerts,
            "mitigation_strategy_recommended": (
                self.bias_config["mitigation_strategy"]
                if is_any_bias_detected
                else "None"
            ),
        }

        self.bias_metrics_history.append(result)
        if is_any_bias_detected:
            logger.warning(f"Bias Detection Alert: {', '.join(bias_alerts)}")
        else:
            logger.info("No significant bias detected in recent decisions.")

        return result

    def mitigate_bias(self, rl_agent_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggests or applies mitigation strategies to RL agent parameters.
        This would typically modify reward functions, cost functions, or constraints.
        """
        if not self.bias_config["enabled"]:
            return rl_agent_params

        latest_bias_check = (
            self.bias_metrics_history[-1] if self.bias_metrics_history else None
        )
        if not latest_bias_check or not latest_bias_check["bias_detected"]:
            logger.info("No active bias detected, no mitigation applied.")
            return rl_agent_params

        mitigation_strategy = self.bias_config["mitigation_strategy"]
        modified_params = rl_agent_params.copy()

        logger.info(f"Applying bias mitigation strategy: {mitigation_strategy}")

        if mitigation_strategy == "re_weight_costs":
            # Example: Increase penalty for assigning low-payout orders to junior drivers
            # or increase incentive for high-payout orders for them.
            if "payout_disparity_ratio" in latest_bias_check["bias_scores"]:
                # Heuristic: If junior drivers are getting low payouts, adjust their specific reward component
                if (
                    latest_bias_check["bias_scores"].get(
                        "junior_drivers_payout_disparity_ratio", 1.0
                    )
                    < self.bias_config["fairness_thresholds"]["payout_disparity_ratio"]
                ):
                    logger.debug(
                        "Adjusting reward weights to address junior driver payout bias."
                    )
                    # Assuming a reward component exists like `payout_component_junior_driver_multiplier`
                    # For demo, just modify a dummy parameter.
                    modified_params["reward_functions"][
                        "payout_component_junior_driver_multiplier"
                    ] = (
                        modified_params["reward_functions"].get(
                            "payout_component_junior_driver_multiplier", 1.0
                        )
                        * 1.1
                    )

            # Example: Adjust traffic cost for older vehicles if stress disparity is high
            if "stress_exposure_disparity" in latest_bias_check["bias_scores"]:
                if (
                    latest_bias_check["bias_scores"].get(
                        "older_vehicles_stress_exposure_disparity", 0.0
                    )
                    > self.bias_config["fairness_thresholds"][
                        "stress_exposure_disparity"
                    ]
                ):
                    logger.debug(
                        "Adjusting traffic cost for older vehicles to reduce stress exposure."
                    )
                    modified_params["cost_functions"][
                        "traffic_penalty_older_vehicles_multiplier"
                    ] = (
                        modified_params["cost_functions"].get(
                            "traffic_penalty_older_vehicles_multiplier", 1.0
                        )
                        * 1.1
                    )

        elif mitigation_strategy == "constrained_optimization":
            logger.warning(
                "Constrained optimization mitigation requires changes to the RL agent's training loop (e.g., adding fairness constraints). This is a conceptual demonstration."
            )
            # In a real system, this would involve sending flags/constraints to the RL training/inference system.
            modified_params["rl_agent_constraints"] = {
                "fairness_constraint_active": True,
                "target_payout_disparity": 1.0,
            }

        logger.info("Bias mitigation parameters generated/applied.")
        return modified_params


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    ethical_ai:
      bias_management:
        enabled: true
        history_window_size: 100
        fairness_thresholds:
          payout_disparity_ratio: 0.9 # Set slightly higher for easier demo trigger
          high_payout_allocation_disparity: 0.1 # Set lower for easier demo trigger
          stress_exposure_disparity: 0.1
        mitigation_strategy: "re_weight_costs"
        protected_attributes:
          - driver_experience_years
          - region_income_level
          - vehicle_age_years
        disadvantaged_groups:
          driver_experience_years: {"label": "junior_drivers", "condition": "<5"}
          region_income_level: {"label": "low_income_areas", "condition": "low"}
          vehicle_age_years: {"label": "older_vehicles", "condition": ">10"}
"""
            )

    bias_manager = BiasDetectionAndMitigation(config_file)

    # Generate some dummy decisions
    rl_decision_log = MockRLDecisionLog()
    rl_decision_log.generate_dummy_decisions(50)

    # Simulate a scenario where junior drivers get fewer high payouts
    for _ in range(20):
        decision = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "driver_id": random.choice(["driver_0", "driver_1"]),  # Junior drivers
            "current_location_node": random.choice(rl_decision_log.node_ids),
            "driver_experience_years": random.randint(1, 4),  # Junior
            "vehicle_age_years": random.randint(1, 5),
            "is_urban_area": True,
            "chosen_action": "assign_low_payout",
            "payout_offered": random.uniform(10, 40),
            "estimated_travel_time_minutes": random.uniform(10, 30),
            "predicted_traffic_factor": random.uniform(0.8, 1.2),
            "region_income_level": random.choice(["low", "medium"]),
            "demand_level_at_node": random.uniform(0.1, 0.5),
        }
        rl_decision_log.add_decision(decision)

    # Simulate older vehicles getting more high-traffic routes
    for _ in range(15):
        decision = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "driver_id": random.choice(["driver_2", "driver_3"]),
            "current_location_node": random.choice(rl_decision_log.node_ids),
            "driver_experience_years": random.randint(6, 10),
            "vehicle_age_years": random.randint(11, 15),  # Older vehicle
            "is_urban_area": True,
            "chosen_action": "move_to_demand",
            "payout_offered": random.uniform(50, 100),
            "estimated_travel_time_minutes": random.uniform(40, 70),  # Longer travel
            "predicted_traffic_factor": random.uniform(1.3, 1.8),  # High traffic
            "region_income_level": random.choice(["medium", "high"]),
            "demand_level_at_node": random.uniform(0.6, 1.0),
        }
        rl_decision_log.add_decision(decision)

    decisions_df = pd.DataFrame(list(rl_decision_log.log))

    print("--- Detecting Bias ---")
    bias_report = bias_manager.detect_bias(decisions_df)
    print(json.dumps(bias_report, indent=2, ensure_ascii=False))

    # Simulate RL agent parameters
    current_rl_agent_params = {
        "reward_functions": {
            "payout_component": 1.0,
            "payout_component_junior_driver_multiplier": 1.0,  # Will be modified
            "stress_penalty_component": 1.0,
        },
        "cost_functions": {
            "traffic_penalty": 1.0,
            "traffic_penalty_older_vehicles_multiplier": 1.0,  # Will be modified
        },
        "rl_agent_constraints": {},
    }

    print("\n--- Applying Bias Mitigation ---")
    mitigated_params = bias_manager.mitigate_bias(current_rl_agent_params)
    print("Mitigated RL Agent Parameters:")
    print(json.dumps(mitigated_params, indent=2, ensure_ascii=False))

    print("\nBias Detection and Mitigation demo complete.")
