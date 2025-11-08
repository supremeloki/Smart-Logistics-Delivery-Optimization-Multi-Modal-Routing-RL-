import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import yaml
import logging
import argparse
import os
import random
import networkx as nx
import osmnx as ox

from feature_forge.graph_embedding_features import GraphConvolutionalNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNTrainer:
    def __init__(self, config_path, graph_path, model_output_path):
        self.config = self._load_config(config_path)
        self.graph_path = graph_path
        self.model_output_path = model_output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.graph = self._load_graph()
        self.pyg_data, self.node_map = self._preprocess_graph_for_pyg(self.graph)
        self.pyg_data = self.pyg_data.to(self.device)

        num_node_features = self.pyg_data.x.shape[1]
        hidden_channels = self.config["training_params"]["model"].get(
            "hidden_channels_gnn", 64
        )
        num_output_features = self.config["multi_agent_config"]["policies"][
            "driver_policy"
        ]["obs_space"][
            0
        ]  # Use part of observation space as target

        self.model = GraphConvolutionalNetwork(
            num_node_features, hidden_channels, num_output_features
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["training_params"].get("gnn_lr", 0.001),
        )
        self.loss_fn = (
            nn.MSELoss()
        )  # Assuming a regression task for embeddings or features

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_graph(self):
        graph = nx.read_gml(self.graph_path)
        # Ensure 'x', 'y' for Haversine, and 'closeness_centrality'
        for n, data in graph.nodes(data=True):
            if "x" not in data:
                data["x"] = random.uniform(51.2, 51.7)
            if "y" not in data:
                data["y"] = random.uniform(35.5, 35.9)
            if "closeness_centrality" not in data:
                data["closeness_centrality"] = random.random()
        return graph

    def _preprocess_graph_for_pyg(self, graph):
        node_features = []
        node_map = {node_id: i for i, node_id in enumerate(graph.nodes())}

        for node_id, data in graph.nodes(data=True):
            features = [
                data.get("x", 0.0),
                data.get("y", 0.0),
                graph.degree(node_id),
                data.get("closeness_centrality", 0.0),
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        edge_indices = []
        for u, v, _ in graph.edges(keys=True):
            edge_indices.append([node_map[u], node_map[v]])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index), node_map

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Dummy target for GNN training (e.g., predicting future traffic, or node properties)
            # In a real scenario, this would come from labels.
            # For pure embedding, we might use a self-supervised task or just use the embeddings directly.
            # Here, let's assume it's predicting a set of 'target' features
            target = torch.randn(
                self.pyg_data.x.shape[0], self.model.conv2.out_channels
            ).to(self.device)

            out = self.model(self.pyg_data.x, self.pyg_data.edge_index)
            loss = self.loss_fn(out, target)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"GNN Epoch: {epoch+1:03d}, Loss: {loss:.4f}")

        logger.info("GNN training complete.")
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_output_path)
        logger.info(f"GNN model saved to {self.model_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="conf/rl_agent_params.yaml",
        help="Path to RL agent config file (used for GNN params)",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="data_nexus/road_network_graph/preprocessed_tehran_graph.gml",
        help="Path to preprocessed graph",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="rl_model_registry/gcn_model.pth",
        help="Path to save trained GNN model",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    args = parser.parse_args()

    # Ensure graph file exists, generate a dummy one if not
    if not os.path.exists(args.graph_path):
        logger.warning(
            f"Graph file not found at {args.graph_path}. Generating a temporary small graph."
        )

        G_temp = ox.graph_from_point((35.7, 51.4), dist=500, network_type="drive")
        G_temp = ox.add_edge_speeds(G_temp)
        G_temp = ox.add_edge_travel_times(G_temp)
        for n, data in G_temp.nodes(data=True):
            data["x"] = G_temp.nodes[n]["x"]
            data["y"] = G_temp.nodes[n]["y"]
            data["closeness_centrality"] = random.random()
        os.makedirs(os.path.dirname(args.graph_path), exist_ok=True)
        nx.write_gml(G_temp, args.graph_path)

    trainer = GNNTrainer(args.config, args.graph_path, args.model_output)
    trainer.train(args.epochs)
