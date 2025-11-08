from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import PolicySpec


def map_agent_to_policy(agent_id, episode, worker, **kwargs):
    if "driver" in agent_id:
        return "driver_policy"
    elif "dispatcher" in agent_id:
        return "fleet_manager_policy"
    return "default_policy"


class MultiAgentPolicyContainer:
    def __init__(self, config):
        self.config = config
        self.trainer = PPO(config=self.config)

    def get_policies(self):
        return {
            "driver_policy": PolicySpec(
                observation_space=self.config["multi_agent_config"]["policies"][
                    "driver_policy"
                ]["obs_space"],
                action_space=self.config["multi_agent_config"]["policies"][
                    "driver_policy"
                ]["action_space"],
            ),
            "fleet_manager_policy": PolicySpec(
                observation_space=self.config["multi_agent_config"]["policies"][
                    "fleet_manager_policy"
                ]["obs_space"],
                action_space=self.config["multi_agent_config"]["policies"][
                    "fleet_manager_policy"
                ]["action_space"],
            ),
        }

    def infer_actions(self, observations):
        actions = {}
        for agent_id, obs in observations.items():
            policy_id = map_agent_to_policy(agent_id, None, None)
            agent_action = self.trainer.compute_single_action(
                obs, policy_id=policy_id, explore=False
            )
            actions[agent_id] = agent_action
        return actions

    def load_checkpoint(self, checkpoint_path):
        self.trainer.restore(checkpoint_path)

    def save_checkpoint(self, checkpoint_dir):
        return self.trainer.save(checkpoint_dir)
