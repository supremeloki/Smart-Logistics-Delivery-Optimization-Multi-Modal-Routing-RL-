import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import os
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Mock Driver Behavior Model (e.g., predicting next move or efficiency)
class DriverBehaviorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class FederatedLearningClient:
    def __init__(
        self,
        driver_id: str,
        config_path="conf/environments/prod.yaml",
        environment="dev",
    ):
        self.driver_id = driver_id
        self.config = self._load_config(config_path)
        self.fl_config = self.config["environments"][environment]["federated_learning"]
        self.model = DriverBehaviorModel(
            input_dim=self.fl_config["model_params"]["input_dim"],
            hidden_dim=self.fl_config["model_params"]["hidden_dim"],
            output_dim=self.fl_config["model_params"]["output_dim"],
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.fl_config["learning_rate"]
        )
        self.criterion = nn.MSELoss()  # Example loss for regression task

        # Placeholder for local data collected by this driver
        # In a real scenario, this would come from the driver's vehicle data (telemetry, route choices, performance)
        self.local_dataset = self._generate_dummy_local_data()
        self.data_loader = DataLoader(
            self.local_dataset, batch_size=self.fl_config["batch_size"], shuffle=True
        )

        self.local_model_path = os.path.join(
            self.fl_config["local_model_dir"], f"{self.driver_id}_local_model.pth"
        )
        os.makedirs(self.fl_config["local_model_dir"], exist_ok=True)
        self._load_local_model()
        logger.info(
            f"Federated Learning Client for Driver {self.driver_id} initialized."
        )

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    federated_learning:
      enabled: true
      local_model_dir: rl_model_registry/federated_models
      server_endpoint: http://localhost:8002/federated_update # Central server endpoint
      learning_rate: 0.001
      batch_size: 16
      local_epochs: 5
      model_params:
        input_dim: 64
        hidden_dim: 128
        output_dim: 10
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _generate_dummy_local_data(self):
        # Simulate local driver data
        num_samples = random.randint(50, 200)
        X = torch.randn(num_samples, self.fl_config["model_params"]["input_dim"])
        y = torch.randn(num_samples, self.fl_config["model_params"]["output_dim"])
        return TensorDataset(X, y)

    def _load_local_model(self):
        if os.path.exists(self.local_model_path):
            self.model.load_state_dict(torch.load(self.local_model_path))
            logger.info(
                f"Loaded local model for {self.driver_id} from {self.local_model_path}"
            )
        else:
            logger.info(
                f"No existing local model found for {self.driver_id}. Initializing new model."
            )

    def _save_local_model(self):
        torch.save(self.model.state_dict(), self.local_model_path)
        logger.info(
            f"Saved local model for {self.driver_id} to {self.local_model_path}"
        )

    def train_local_model(self):
        self.model.train()
        total_loss = 0.0
        for epoch in range(self.fl_config["local_epochs"]):
            for inputs, targets in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        logger.info(
            f"Driver {self.driver_id} finished local training. Average loss: {total_loss / len(self.data_loader):.4f}"
        )
        self._save_local_model()
        return self.model.state_dict()  # Return updated weights

    def get_model_weights(self):
        return self.model.state_dict()

    def update_model_with_global_weights(self, global_weights):
        self.model.load_state_dict(global_weights)
        logger.info(f"Driver {self.driver_id} updated model with global weights.")
        self._save_local_model()  # Save updated global model locally


if __name__ == "__main__":
    # Ensure dummy config for dev environment
    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    federated_learning:
      enabled: true
      local_model_dir: rl_model_registry/federated_models
      server_endpoint: http://localhost:8002/federated_update
      learning_rate: 0.001
      batch_size: 16
      local_epochs: 2
      model_params:
        input_dim: 64
        hidden_dim: 128
        output_dim: 10
"""
            )

    # --- Demonstrate a single client's lifecycle ---
    client_id = "driver_alpha_001"
    fl_client = FederatedLearningClient(client_id, config_file)

    print(f"\n--- Driver {client_id} performing local training ---")
    local_weights = fl_client.train_local_model()

    # Simulate receiving global weights from a server (for demonstration, just re-use local_weights for now)
    print(f"\n--- Driver {client_id} receiving and updating with global weights ---")
    simulated_global_weights = {
        k: v * 0.9 + fl_client.model.state_dict()[k] * 0.1
        for k, v in local_weights.items()
    }  # Simple averaging effect
    fl_client.update_model_with_global_weights(simulated_global_weights)

    # You could then perform inference with the updated model
    dummy_input = torch.randn(1, fl_client.fl_config["model_params"]["input_dim"])
    output = fl_client.model(dummy_input)
    print(f"\n--- Driver {client_id} performing inference with updated model ---")
    print(f"Dummy input: {dummy_input.detach().numpy().flatten()[:5]}...")
    print(f"Model output: {output.detach().numpy().flatten()[:5]}...")

    print(f"\nFederated Learning Client for {client_id} demo complete.")
