from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from torch import nn
import numpy as np


class ActorCriticNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.shared_layer_dims = model_config.get("fcnet_hiddens", [256, 256])
        self.obs_size = self._get_size_from_obs_space(obs_space)

        self.shared_net = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs=(
                self.shared_layer_dims[-1] if self.shared_layer_dims else self.obs_size
            ),
            model_config={"fcnet_hiddens": self.shared_layer_dims},
            name="shared_net_fc",
        )

        self.actor_head = nn.Linear(
            self.shared_layer_dims[-1] if self.shared_layer_dims else self.obs_size,
            num_outputs,
        )
        self.critic_head = nn.Linear(
            self.shared_layer_dims[-1] if self.shared_layer_dims else self.obs_size, 1
        )

        self._value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        feature_embedding, _ = self.shared_net(input_dict)
        logits = self.actor_head(feature_embedding)
        self._value = self.critic_head(feature_embedding).squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value

    def _get_size_from_obs_space(self, obs_space):
        if hasattr(obs_space, "shape"):
            return int(np.product(obs_space.shape))
        return 0  # Or handle other types if needed
