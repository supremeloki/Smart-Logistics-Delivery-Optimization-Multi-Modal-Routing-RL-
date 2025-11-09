import ray
import yaml
import logging
import argparse
import os
import networkx as nx
import random
import pandas as pd
import numpy as np

import sys
import os

# Add the parent directory to the Python path to enable relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_nexus.simulation_scenarios.simpy_delivery_environment import SimpyDeliveryEnvironment
from routing.astar_optimization_logic import AStarRouting
from learning.multi_agent_rl_policy import MultiAgentPolicyContainer
from feature_forge.graph_embedding_features import (
    GraphEmbeddingFeatures,
)

# Import lono_libs components for evaluation
try:
    # Try to import lono_libs - it's expected to be installed or available in path
    from lono_libs import Evaluator, ScoreAggregator, get_all_metrics
    from lono_libs.classification import Accuracy, F1Score, Precision, Recall, ROCAUC, LogLoss
    from lono_libs.regression import MeanSquaredError, MeanAbsoluteError, R2Score, MeanAbsolutePercentageError
    from lono_libs import VisualizationGenerator, ReportingGenerator, SummaryReportGenerator
    LONO_AVAILABLE = True
except ImportError:
    LONO_AVAILABLE = False
    logging.warning("lono_libs not available. Using fallback evaluation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLPerformanceMetric:
    """Custom metric class for RL performance evaluation"""

    def __init__(self, name, higher_better=True, weight=1.0, target_score=None):
        self.name = name
        self.is_higher_better = higher_better
        self.weight = weight
        self.target_score = target_score

    def calculate(self, y_true, y_pred):
        """Calculate metric value"""
        # For RL, y_true and y_pred are arrays of episode metrics
        if self.name == "MeanDeliveredOrders":
            return np.mean(y_pred) if len(y_pred) > 0 else 0.0
        elif self.name == "Efficiency":
            # Custom efficiency metric: orders per distance
            return np.mean(y_pred) if len(y_pred) > 0 else 0.0
        elif self.name == "Reward":
            return np.mean(y_pred) if len(y_pred) > 0 else 0.0
        else:
            return np.mean(y_pred) if len(y_pred) > 0 else 0.0

    def calculate_from_proba(self, y_true, y_pred_proba):
        """Not used for RL metrics"""
        raise NotImplementedError


class LonoRLBenchmarker:
    def __init__(
        self, config_path, scenario_path, output_dir="experiment_lab/rl_experiment_runs"
    ):
        self.config = self._load_config(config_path)
        self.scenario_path = scenario_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.router = self._initialize_router()
        self.gnn_embedder = self._initialize_gnn_embedder()
        self.rl_inferers = {}  # Store different RL agents/policies for benchmarking

        # Initialize lono_libs evaluator if available
        if LONO_AVAILABLE:
            self._setup_lono_evaluator()
        else:
            logger.warning("Using basic evaluation without lono_libs metrics")

    def _setup_lono_evaluator(self):
        """Setup lono_libs evaluator with custom RL metrics"""
        rl_metrics = [
            RLPerformanceMetric("MeanDeliveredOrders", higher_better=True, weight=0.4),
            RLPerformanceMetric("Efficiency", higher_better=True, weight=0.3),
            RLPerformanceMetric("Reward", higher_better=True, weight=0.3),
        ]

        # Add some standard regression metrics for comparison
        try:
            rl_metrics.extend([
                MeanSquaredError(),
                MeanAbsoluteError(),
                R2Score()
            ])
        except:
            pass

        self.evaluator = Evaluator(metrics=rl_metrics)
        self.score_aggregator = ScoreAggregator({m.name: m for m in rl_metrics})

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _initialize_router(self):
        graph_path = self.config["osm_processing"]["output_path"]
        if not os.path.exists(graph_path):
            logger.error(
                "Graph file not found. Benchmarking cannot proceed without a valid graph."
            )
            raise FileNotFoundError(f"Graph file missing: {graph_path}")
        graph = nx.read_gml(graph_path)
        return AStarRouting(graph, "conf/routing_engine_config.yaml")

    def _initialize_gnn_embedder(self):
        graph_path = self.config["osm_processing"]["output_path"]
        gnn_config_path = "conf/rl_agent_params.yaml"  # Using this for model params
        embedder = GraphEmbeddingFeatures(gnn_config_path, graph_path)
        gnn_model_path = self.config.get("gnn_model_path", "rl_model_registry/gcn_model.pth")
        if os.path.exists(gnn_model_path):
            try:
                embedder.load_model_weights(gnn_model_path)
            except:
                logger.warning(
                    f"Failed to load GNN model from {gnn_model_path}. Using untrained GNN for embeddings."
                )
        else:
            logger.warning(
                f"GNN model not found at {gnn_model_path}. Using untrained GNN for embeddings."
            )
        return embedder

    def add_rl_agent(self, agent_id, checkpoint_path=None, rl_config_override=None):
        if rl_config_override:
            agent_config = rl_config_override
        else:
            agent_config = self.config["rl_agent_params"]

        # Ensure multi_agent_config is present in rl_config
        if "multi_agent_config" not in agent_config:
            raise ValueError("RL agent config must contain 'multi_agent_config'.")

        # For benchmarking, we load the actual policy weights into a MultiAgentPolicyContainer
        inferer = MultiAgentPolicyContainer(agent_config)
        if checkpoint_path and os.path.exists(checkpoint_path):
            # This is a bit of a hack as MultiAgentPolicyContainer doesn't directly restore Trainer checkpoints
            # In a full RLlib setup, you would create a PPOConfig and build() then restore().
            # For a dedicated inference class (like TritonAdapter), it would abstract this.
            # Here, we simulate loading for direct use.
            # A more robust solution would be to load the policy directly:
            # from ray.rllib.policy.policy import Policy
            # policy = Policy.from_checkpoint({"path": checkpoint_path, "policy_id": "driver_policy"})
            logger.warning(
                "RLlib checkpoint loading for direct inference is complex. Using dummy inferer for now."
            )
            # Assume checkpoint is a directory containing `policies/`
            # inferer.load_checkpoint(checkpoint_path)
        else:
            logger.warning(
                f"No checkpoint found for agent {agent_id}. Agent will behave randomly."
            )

        self.rl_inferers[agent_id] = inferer

    def run_benchmark(self, agent_id, num_episodes=5, sim_duration=600):
        if agent_id not in self.rl_inferers:
            raise ValueError(f"Agent '{agent_id}' not added to benchmarker.")

        all_episode_metrics = []
        rl_inferer = self.rl_inferers[agent_id]

        for episode in range(num_episodes):
            logger.info(
                f"Running benchmark for agent '{agent_id}', Episode {episode + 1}/{num_episodes}"
            )
            # Each episode gets a fresh simulation environment
            sim_env = SimpyDeliveryEnvironment(
                "conf/environments/dev.yaml",  # Use dev config for simulation
                self.scenario_path,
                self.router,
                rl_inferer,  # Pass the specific RL inferer
            )
            metrics_df = sim_env.run_simulation(until=sim_duration)

            # Aggregate episode metrics
            final_metrics = metrics_df.iloc[-1].to_dict()
            final_metrics["agent_id"] = agent_id
            final_metrics["episode"] = episode
            all_episode_metrics.append(final_metrics)

        # Convert to DataFrame and prepare for evaluation
        results_df = pd.DataFrame(all_episode_metrics)

        if LONO_AVAILABLE and hasattr(self, 'evaluator'):
            # Use lono_libs for advanced evaluation
            evaluation_results = self._evaluate_with_lono(results_df, agent_id)
            return results_df, evaluation_results
        else:
            return results_df, None

    def _evaluate_with_lono(self, results_df, agent_id):
        """Evaluate results using lono_libs evaluator"""
        # Prepare data for lono_libs evaluation
        # For RL, we treat episode metrics as regression targets
        metric_columns = ["num_delivered_orders", "total_driver_distance", "total_driver_time", "num_pending_orders"]

        evaluation_data = []
        for idx, row in results_df.iterrows():
            for metric_col in metric_columns:
                if metric_col in row:
                    # Create mock true values (could be baseline or target values)
                    y_true = np.array([row[metric_col]])  # Using the actual value as "true" for self-evaluation
                    y_pred = np.array([row[metric_col]])

                    evaluation_data.append({
                        'model_name': f"{agent_id}_episode_{idx}",
                        'model_type': 'RL_Agent',
                        'y_true': y_true,
                        'y_pred_train': y_pred,  # Same for train/test in this context
                        'y_pred_test': y_pred,
                        'y_pred_proba_train': None,
                        'y_pred_proba_test': None,
                    })

        if evaluation_data:
            full_results_df, best_models_by_metric = self.evaluator.evaluate_models(evaluation_data)
            return {
                'full_results': full_results_df,
                'best_models': best_models_by_metric,
                'weighted_scores': self._calculate_weighted_scores(full_results_df, agent_id)
            }
        return None

    def _calculate_weighted_scores(self, full_results_df, agent_id):
        """Calculate weighted scores for each episode"""
        weighted_scores = {}
        for model_name in full_results_df['model_name'].unique():
            if agent_id in model_name:
                model_results = full_results_df[full_results_df['model_name'] == model_name].to_dict('records')
                weighted_score = self.score_aggregator.calculate_weighted_average(model_results)
                episode_num = int(model_name.split('_episode_')[-1])
                weighted_scores[episode_num] = weighted_score
        return weighted_scores

    def compare_results(self, results_dfs: dict):
        """Compare results across agents using lono_libs if available"""
        # results_dfs is a dict of {agent_id: (DataFrame_of_metrics, evaluation_results)}

        # Extract basic metrics comparison
        combined_df = pd.concat([df[0].assign(agent_id=agent_id) for agent_id, df in results_dfs.items()], ignore_index=True)

        logger.info("\n--- Basic RL Performance Summary ---")
        summary = combined_df.groupby("agent_id").agg(
            {
                "num_delivered_orders": ["mean", "std"],
                "total_driver_distance": ["mean", "std"],
                "total_driver_time": ["mean", "std"],
                "num_pending_orders": "mean",
            }
        )
        logger.info(summary)

        # Advanced evaluation with lono_libs if available
        if LONO_AVAILABLE and any(df[1] is not None for df in results_dfs.values()):
            logger.info("\n--- Lono_libs Advanced Evaluation ---")
            self._generate_advanced_reports(results_dfs)

        return summary

    def _generate_advanced_reports(self, results_dfs):
        """Generate advanced reports using lono_libs reporting tools"""
        try:
            # Combine all evaluation results
            all_evaluation_results = []
            for agent_id, (basic_df, eval_results) in results_dfs.items():
                if eval_results and 'full_results' in eval_results:
                    df = eval_results['full_results'].copy()
                    df['agent_id'] = agent_id
                    all_evaluation_results.append(df)

            if all_evaluation_results:
                combined_eval_df = pd.concat(all_evaluation_results, ignore_index=True)

                # Generate reports
                if hasattr(self, 'evaluator'):
                    reporter = ReportingGenerator()
                    reporter.print_performance_report(combined_eval_df)

                    best_models_by_metric = {}
                    for agent_id, (basic_df, eval_results) in results_dfs.items():
                        if eval_results and 'best_models' in eval_results:
                            for metric, df in eval_results['best_models'].items():
                                if metric not in best_models_by_metric:
                                    best_models_by_metric[metric] = []
                                df = df.copy()
                                df['agent_id'] = agent_id
                                best_models_by_metric[metric].append(df)

                    # Find overall best for each metric
                    overall_best = {}
                    for metric, dfs in best_models_by_metric.items():
                        combined = pd.concat(dfs, ignore_index=True)
                        if not combined.empty:
                            if combined['is_higher_better'].iloc[0]:
                                overall_best[metric] = combined.loc[combined['Testing Score'].idxmax()].to_frame().T
                            else:
                                overall_best[metric] = combined.loc[combined['Testing Score'].idxmin()].to_frame().T

                    reporter.print_best_performance_summary(overall_best)

                    # Generate visualizations if requested
                    viz_dir = os.path.join(self.output_dir, "visualizations")
                    viz_generator = VisualizationGenerator(combined_eval_df, overall_best, output_dir=viz_dir)
                    viz_generator.generate_all_plots()

        except Exception as e:
            logger.error(f"Error generating advanced reports: {e}")
