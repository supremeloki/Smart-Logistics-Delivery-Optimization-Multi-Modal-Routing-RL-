import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import yaml
import logging
import networkx as nx
import numpy as np
import osmnx as ox
import os
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphEmbeddingFeatures:
    def __init__(self, config_path, graph_path):
        self.config = self._load_config(config_path)
        self.graph = self._load_graph(graph_path)
        self.model = self._initialize_model()

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_graph(self, graph_path):
        return nx.read_gml(graph_path)

    def _initialize_model(self):
        # Dummy feature dimensions for initialization
        num_node_features = 4  # Example: x, y, degree, centrality
        hidden_channels = 64
        num_output_features = 32  # Embedding dimension
        model = GraphConvolutionalNetwork(
            num_node_features, hidden_channels, num_output_features
        )
        return model

    def preprocess_graph_for_pyg(self, graph):
        node_features = []
        node_map = {node_id: i for i, node_id in enumerate(graph.nodes())}

        for node_id, data in graph.nodes(data=True):
            features = [
                data.get("x", 0.0),
                data.get("y", 0.0),
                graph.degree(node_id),
                data.get(
                    "closeness_centrality", 0.0
                ),  # Assume centrality is precalculated
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        edge_indices = []
        for u, v, _ in graph.edges(keys=True):  # For multigraph
            edge_indices.append([node_map[u], node_map[v]])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index), node_map

    def generate_embeddings(self, pyg_data):
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(pyg_data.x, pyg_data.edge_index)
        return embeddings

    def load_model_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        logger.info(f"GNN model weights loaded from {model_path}")

    def get_node_embedding(self, node_id, embeddings_tensor, node_map):
        if node_id in node_map:
            pyg_idx = node_map[node_id]
            return embeddings_tensor[pyg_idx].detach().numpy()
        return np.zeros(
            embeddings_tensor.shape[1]
        )  # Return zero vector if node not found


if __name__ == "__main__":
    # Example usage:
    # 1. Ensure 'data_nexus/road_network_graph/preprocessed_tehran_graph.gml' exists (run osmnx_processor.py)
    # 2. Assume a dummy GCN model exists at 'model_checkpoints/gcn_model.pth'

    # Create dummy config and model path
    dummy_config_path = "conf/rl_agent_params.yaml"  # Any config will do for loading
    dummy_model_path = "model_checkpoints/gcn_model.pth"

    # Create dummy model file for demonstration if it doesn't exist
    if not os.path.exists("model_checkpoints"):
        os.makedirs("model_checkpoints")

    try:
        # Generate a temporary small graph if main one isn't ready
        temp_graph_path = "data_nexus/road_network_graph/temp_graph.gml"
        if not os.path.exists(temp_graph_path):
            G_temp = ox.graph_from_point((35.7, 51.4), dist=1000, network_type="drive")
            G_temp = ox.add_edge_speeds(G_temp)
            G_temp = ox.add_edge_travel_times(G_temp)
            for n, data in G_temp.nodes(data=True):
                data["closeness_centrality"] = random.random()
            nx.write_gml(G_temp, temp_graph_path)

        graph_embedding_tool = GraphEmbeddingFeatures(
            dummy_config_path, temp_graph_path
        )

        # Save dummy model weights
        torch.save(graph_embedding_tool.model.state_dict(), dummy_model_path)

        # Load weights for real usage
        graph_embedding_tool.load_model_weights(dummy_model_path)

        pyg_data, node_map = graph_embedding_tool.preprocess_graph_for_pyg(
            graph_embedding_tool.graph
        )
        embeddings = graph_embedding_tool.generate_embeddings(pyg_data)

        logger.info(f"Generated node embeddings with shape: {embeddings.shape}")

        # Get an embedding for a specific node
        sample_node_id = list(graph_embedding_tool.graph.nodes())[0]
        node_embedding = graph_embedding_tool.get_node_embedding(
            sample_node_id, embeddings, node_map
        )
        logger.info(f"Embedding for node {sample_node_id}: {node_embedding[:5]}...")

    except FileNotFoundError:
        logger.error(
            "Graph file not found. Please ensure it exists or run osmnx_processor.py."
        )
    except Exception as e:
        logger.error(f"Error during graph embedding example: {e}")
