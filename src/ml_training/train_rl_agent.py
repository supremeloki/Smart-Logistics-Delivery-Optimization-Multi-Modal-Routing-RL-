import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import yaml
import logging
import argparse
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym
import networkx as nx

from data_nexus.simulation_scenarios.simpy_delivery_environment import SimpyDeliveryEnvironment
from routing.astar_optimization_logic import AStarRouting
from learning.multi_agent_rl_policy import MultiAgentPolicyContainer
from learning.policy_networks.actor_critic import ActorCriticNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dummy Env Wrapper for RLlib compatibility
class RLLibSimpyEnv(gym.Env):
    def __init__(self, env_config):
        self.simpy_env = SimpyDeliveryEnvironment(
            env_config["config_path"],
            env_config["scenario_path"],
            env_config["router_instance"],
            env_config[
                "rl_agent_inferer"
            ],  # This would be `self` in a real setup for RLlib
        )
        self.action_space = gym.spaces.Discrete(
            env_config.get("action_space_size", 5)
        )  # Example
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(env_config.get("obs_space_size", 128),)
        )  # Example
        self.current_state = self.observation_space.sample()

    def reset(self, seed=None, options=None):
        self.simpy_env = SimpyDeliveryEnvironment(
            self.simpy_env.config_path,
            self.simpy_env.scenario_path,
            self.simpy_env.router,
            self.simpy_env.rl_agent_inferer,  # Needs to be updated or passed dynamically
        )  # Reset the entire SimPy environment
        self.current_state = self.observation_space.sample()
        return self.current_state, {}

    def step(self, action):
        # In a real multi-agent scenario, this would be more complex
        # RLlib handles multi-agent through its API directly, usually `compute_actions` for individual agents
        # For a centralized agent, the SimPy env would expose the state for the fleet manager.
        # This is a highly simplified step.

        # Simulate environment step
        self.simpy_env.env.step()

        observation = self.observation_space.sample()
        reward = np.random.rand() * 10 - 5  # Dummy reward
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


def env_creator(env_config):
    return RLLibSimpyEnv(env_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="conf/rl_agent_params.yaml",
        help="Path to RL agent config file",
    )
    parser.add_argument(
        "--env-scenario",
        type=str,
        default="data_nexus/simulation_scenarios/tehran_fleet_scenario.pkl",
        help="Path to simulation scenario file",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=10, help="Number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="rl_model_registry",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        rl_config = yaml.safe_load(f)

    ray.init(ignore_reinit_error=True, logging_level=logging.INFO)

    register_env("simpy_logistics_env", env_creator)

    # Initialize router and dummy inferer for environment creation
    graph_path = "data_nexus/road_network_graph/preprocessed_tehran_graph.gml"
    router_instance = AStarRouting(
        nx.read_gml(graph_path), "conf/routing_engine_config.yaml"
    )
    dummy_inferer = MultiAgentPolicyContainer(
        rl_config
    )  # Placeholder, actual inferer would load trained policy

    env_config_for_rllib = {
        "config_path": "conf/environments/dev.yaml",
        "scenario_path": args.env_scenario,
        "router_instance": router_instance,
        "rl_agent_inferer": dummy_inferer,
        "action_space_size": 5,  # Must match policy_spec
        "obs_space_size": 128,  # Must match policy_spec
    }

    config = (
        PPOConfig()
        .environment("simpy_logistics_env", env_config=env_config_for_rllib)
        .framework("torch")
        .rollouts(num_rollout_workers=rl_config["training_params"]["num_workers"])
        .training(
            gamma=rl_config["training_params"]["gamma"],
            lr=rl_config["training_params"]["lr"],
            train_batch_size=rl_config["training_params"]["train_batch_size"],
            sgd_minibatch_size=rl_config["training_params"]["sgd_minibatch_size"],
            model={
                "fcnet_hiddens": rl_config["training_params"]["model"]["fcnet_hiddens"],
                "vf_share_layers": rl_config["training_params"]["model"][
                    "vf_share_layers"
                ],
            },
        )
        .exploration(
            exploration_config=rl_config["training_params"]["exploration_config"]
        )
        .multi_agent(
            policies={
                "driver_policy": (
                    ActorCriticNetwork,
                    gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=rl_config["multi_agent_config"]["policies"][
                            "driver_policy"
                        ]["obs_space"],
                    ),
                    gym.spaces.Discrete(
                        rl_config["multi_agent_config"]["policies"]["driver_policy"][
                            "action_space"
                        ][0]
                    ),
                    {},
                ),
                "fleet_manager_policy": (
                    ActorCriticNetwork,
                    gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=rl_config["multi_agent_config"]["policies"][
                            "fleet_manager_policy"
                        ]["obs_space"],
                    ),
                    gym.spaces.Discrete(
                        rl_config["multi_agent_config"]["policies"][
                            "fleet_manager_policy"
                        ]["action_space"][0]
                    ),
                    {},
                ),
            },
            policy_mapping_fn=MultiAgentPolicyContainer.map_agent_to_policy,
        )
    )

    algo = config.build()

    logger.info(f"Starting RL agent training for {args.num_iterations} iterations...")
    for i in range(args.num_iterations):
        result = algo.train()
        logger.info(
            f"Iteration {i}: {result['episode_reward_mean']:.2f} mean reward, {result['episodes_this_iter']} episodes."
        )
        if i % 5 == 0:
            checkpoint = algo.save(args.checkpoint_dir)
            logger.info(f"Checkpoint saved at {checkpoint}")

    ray.shutdown()
    logger.info("RL agent training completed.")
