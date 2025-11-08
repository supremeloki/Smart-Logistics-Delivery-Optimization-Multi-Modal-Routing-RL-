import numpy as np
import torch
import torch.nn as nn


class ValueFunctionNetwork(nn.Module):
    def __init__(self, obs_space_dim, hidden_dims, activation_fn=nn.ReLU):
        super().__init__()

        layers = []
        in_dim = obs_space_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_fn())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))  # Output a single value for state-value
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # Final layer for value function is often initialized with a smaller gain
        if len(self.network) > 0 and isinstance(self.network[-1], nn.Linear):
            nn.init.orthogonal_(self.network[-1].weight, gain=0.01)

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        return self.network(obs.float()).squeeze(-1)  # Ensure single scalar output


if __name__ == "__main__":
    # Test the ValueFunctionNetwork
    obs_dim = 128
    hidden = [256, 128, 64]

    value_net = ValueFunctionNetwork(obs_dim, hidden)

    dummy_obs = torch.randn(1, obs_dim)  # Single observation
    value = value_net(dummy_obs)
    print(f"Single observation value: {value.item():.4f}")

    dummy_batch_obs = torch.randn(16, obs_dim)  # Batch of observations
    batch_values = value_net(dummy_batch_obs)
    print(f"Batch observation values shape: {batch_values.shape}")
    print(f"Sample batch values: {batch_values[:5].detach().numpy()}")

    # Test with different activation
    value_net_tanh = ValueFunctionNetwork(obs_dim, hidden, activation_fn=nn.Tanh)
    value_tanh = value_net_tanh(dummy_obs)
    print(f"Value with Tanh activation: {value_tanh.item():.4f}")

    # Test forward pass with numpy array input
    numpy_obs = np.random.rand(obs_dim)
    numpy_value = value_net(numpy_obs)
    print(f"Numpy input value: {numpy_value.item():.4f}")
