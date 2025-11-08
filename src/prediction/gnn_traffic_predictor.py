import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from typing import Dict, Tuple
import datetime
import yaml
import os
import logging
import random
from collections import deque
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Mock Traffic Event Stream for historical data
class MockTrafficStream:
    def __init__(self, graph: nx.MultiDiGraph, history_size=1000):
        self.graph = graph
        self.edges = list(graph.edges(keys=True))
        self.history = deque(
            maxlen=history_size
        )  # Stores (timestamp, u, v, k, actual_travel_time)
        self.sim_time = datetime.datetime.utcnow()

    def generate_event(self):
        u, v, k = random.choice(self.edges)
        base_travel_time = self.graph[u][v][k].get("travel_time", 60)  # Default to 60s
        # Simulate traffic fluctuations and occasional congestion
        factor = random.uniform(0.8, 1.2)
        if random.random() < 0.1:  # 10% chance of heavy congestion
            factor = random.uniform(1.5, 3.0)
        actual_travel_time = max(10, base_travel_time * factor)

        event = {
            "timestamp": self.sim_time,
            "u": u,
            "v": v,
            "key": k,
            "actual_travel_time": actual_travel_time,
        }
        self.history.append(event)
        self.sim_time += datetime.timedelta(
            seconds=random.randint(10, 60)
        )  # Advance time
        return event


# GNN for Traffic Prediction
class GNNTrafficPredictor(nn.Module):
    def __init__(
        self, node_features_dim, edge_features_dim, hidden_dim, output_dim, num_layers=2
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim  # Not directly used by GCNConv, but important for input prep
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # Predicting travel time per edge
        self.num_layers = num_layers

        # Node feature embedding (if raw node features need to be processed)
        self.node_embed = nn.Linear(node_features_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer for edge predictions
        # A simple approach for edge prediction after node embeddings: concatenate features of end nodes
        self.edge_predictor = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + edge_features_dim, hidden_dim
            ),  # Concatenate features of u,v + explicit edge features
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Embed initial node features
        x = self.node_embed(x)
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)  # Apply activation after each GCN layer

        # For edge prediction, we need to extract features for each edge
        # Assume edge_index has shape [2, num_edges]
        # x is the final node embeddings
        row, col = edge_index  # u, v for each edge

        # Concatenate features of source and destination nodes
        edge_node_features = torch.cat([x[row], x[col]], dim=1)

        # Combine with explicit edge features (if any)
        if edge_attr is not None:
            # Ensure edge_attr has compatible dimensions
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(
                    1
                )  # Make it [num_edges, 1] if it's just a single feature
            edge_features_for_pred = torch.cat([edge_node_features, edge_attr], dim=1)
        else:
            edge_features_for_pred = edge_node_features

        # Predict output for each edge
        return self.edge_predictor(edge_features_for_pred).squeeze(
            1
        )  # Squeeze for single value per edge


class TrafficPredictionEngine:
    def __init__(
        self,
        config_path="conf/environments/prod.yaml",
        environment="dev",
        graph_path="data_nexus/road_network_graph/preprocessed_tehran_graph.gml",
    ):
        self.config = self._load_config(config_path)
        self.tp_config = self.config["environments"][environment]["traffic_prediction"]
        self.graph_path = graph_path
        self.graph = self._load_or_generate_graph()

        # Map NetworkX nodes/edges to PyG indices
        self.node_to_pyg_idx = {node: i for i, node in enumerate(self.graph.nodes())}
        self.pyg_idx_to_node = {i: node for i, node in enumerate(self.graph.nodes())}

        self.edge_list = []  # Store (u_idx, v_idx, key_idx) to map back
        self.edge_to_pyg_idx = {}  # Map (u,v,k) to PyG edge_index column

        # Precompute initial PyG Data object components (features will be dynamic)
        self.pyg_data_template = self._create_pyg_data_template()

        self.model = GNNTrafficPredictor(
            node_features_dim=self.tp_config["gnn_model_params"]["node_features_dim"],
            edge_features_dim=self.tp_config["gnn_model_params"]["edge_features_dim"],
            hidden_dim=self.tp_config["gnn_model_params"]["hidden_dim"],
            output_dim=1,  # Predicting travel time
            num_layers=self.tp_config["gnn_model_params"]["num_layers"],
        )
        self.model_optimizer = optim.Adam(
            self.model.parameters(), lr=self.tp_config["learning_rate"]
        )
        self.model_criterion = nn.MSELoss()

        self.model_checkpoint_path = os.path.join(
            self.tp_config["model_dir"], "gnn_traffic_predictor.pth"
        )
        os.makedirs(self.tp_config["model_dir"], exist_ok=True)
        self._load_model()
        logger.info("GNNTrafficPredictor initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    traffic_prediction:
      enabled: true
      model_dir: ml_ops/traffic_prediction_models
      learning_rate: 0.005
      gnn_model_params:
        node_features_dim: 5
        edge_features_dim: 2
        hidden_dim: 64
        num_layers: 2
      history_window_minutes: 60
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_or_generate_graph(self):
        if os.path.exists(self.graph_path):
            return nx.read_gml(self.graph_path)
        else:
            logger.warning(
                f"Graph file not found at {self.graph_path}. Generating a small temporary graph."
            )
            G_temp = nx.MultiDiGraph()
            for i in range(10):
                G_temp.add_node(
                    i,
                    x=random.uniform(51.2, 51.7),
                    y=random.uniform(35.5, 35.9),
                    centrality=random.uniform(0.1, 1.0),
                    type=random.randint(0, 2),
                )
            for i in range(15):
                u, v = random.randint(0, 9), random.randint(0, 9)
                if u != v:
                    G_temp.add_edge(
                        u,
                        v,
                        key=0,
                        travel_time=random.uniform(30, 120),
                        length=random.uniform(100, 500),
                    )
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            nx.write_gml(G_temp, self.graph_path)
            return G_temp

    def _create_pyg_data_template(self):
        # Prepare edge_index for PyG
        edge_indices = []
        raw_edge_features = []  # For attributes like length, free_flow_time
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if u in self.node_to_pyg_idx and v in self.node_to_pyg_idx:
                pyg_u = self.node_to_pyg_idx[u]
                pyg_v = self.node_to_pyg_idx[v]
                edge_indices.append([pyg_u, pyg_v])
                self.edge_list.append((u, v, k))
                self.edge_to_pyg_idx[(u, v, k)] = len(self.edge_list) - 1

                # Extract static edge features (e.g., length, free_flow_time)
                # Ensure it matches `edge_features_dim`
                raw_edge_features.append(
                    [
                        data.get("length", 0.0),
                        data.get("travel_time", 0.0),  # Free-flow travel time
                    ]
                )

        if not edge_indices:
            raise ValueError("Graph has no valid edges for PyG conversion.")

        edge_index_tensor = (
            torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        )
        raw_edge_features_tensor = torch.tensor(raw_edge_features, dtype=torch.float32)

        # Node features (e.g., static node attributes like centrality or type)
        node_features = []
        for pyg_idx in range(len(self.graph.nodes())):
            node_id = self.pyg_idx_to_node[pyg_idx]
            node_data = self.graph.nodes[node_id]
            # Example node features: centrality, type (one-hot or embedded), coordinates
            node_features.append(
                [
                    node_data.get("centrality", 0.5),
                    float(
                        node_data.get("type", 0)
                    ),  # Assuming 'type' is numeric or can be mapped
                    node_data.get("x", 0.0),
                    node_data.get("y", 0.0),
                    random.uniform(0.1, 1.0),  # Dummy dynamic feature slot
                ]
            )
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

        return Data(
            x=node_features_tensor,
            edge_index=edge_index_tensor,
            edge_attr=raw_edge_features_tensor,
        )

    def _load_model(self):
        if os.path.exists(self.model_checkpoint_path):
            self.model.load_state_dict(torch.load(self.model_checkpoint_path))
            logger.info(
                f"Loaded GNN traffic predictor model from {self.model_checkpoint_path}"
            )
        else:
            logger.info(
                "No existing GNN traffic predictor model found. Initializing new model."
            )

    def _save_model(self):
        torch.save(self.model.state_dict(), self.model_checkpoint_path)
        logger.info(
            f"Saved GNN traffic predictor model to {self.model_checkpoint_path}"
        )

    def prepare_pyg_data_from_historical(self, historical_events: deque):
        # Update dynamic node features (e.g., local congestion, current demand at node)
        # This is a key part: how do historical events translate to current node/edge features?
        # For simplicity, let's update a dummy 'dynamic_feature_slot' on nodes based on recent events
        node_x = self.pyg_data_template.x.clone()  # Start with static node features

        # Aggregate recent traffic events per node for a dynamic feature
        node_recent_congestion = {node_id: 0.0 for node_id in self.graph.nodes()}
        recent_cutoff = datetime.datetime.utcnow() - datetime.timedelta(
            minutes=self.tp_config["history_window_minutes"]
        )

        for event in historical_events:
            if event["timestamp"] > recent_cutoff:
                # Higher travel time means more congestion
                node_recent_congestion[event["u"]] += (
                    event["actual_travel_time"]
                    / self.graph[event["u"]][event["v"]][event["key"]].get(
                        "travel_time", 1.0
                    )
                ) - 1.0
                node_recent_congestion[event["v"]] += (
                    event["actual_travel_time"]
                    / self.graph[event["u"]][event["v"]][event["key"]].get(
                        "traffic_factor", 1.0
                    )
                ) - 1.0

        for pyg_idx in range(node_x.shape[0]):
            node_id = self.pyg_idx_to_node[pyg_idx]
            # Update the last feature slot with aggregated congestion
            node_x[pyg_idx, -1] = node_recent_congestion[
                node_id
            ]  # Example: Last feature is dynamic congestion

        # Create ground truth labels for edges that have recent data
        # Only predict for edges that were observed recently
        edge_labels = []
        observed_edge_indices = []

        for event in historical_events:
            if event["timestamp"] > recent_cutoff:
                nx_edge_key = (event["u"], event["v"], event["key"])
                if nx_edge_key in self.edge_to_pyg_idx:
                    pyg_edge_idx = self.edge_to_pyg_idx[nx_edge_key]
                    observed_edge_indices.append(pyg_edge_idx)
                    edge_labels.append(event["actual_travel_time"])

        # Filter self.pyg_data_template.edge_index and edge_attr to only include observed edges
        # This is for training where we have ground truth. For inference, we predict all.

        # For a simplified training example, we will just use the full graph and fill labels where available.
        # This part often requires careful handling of masking and loss calculation.

        # Create a full set of labels (zeros for unobserved)
        full_edge_labels = torch.zeros(len(self.edge_list), dtype=torch.float32)
        for i, idx in enumerate(observed_edge_indices):
            full_edge_labels[idx] = edge_labels[i]

        pyg_data = Data(
            x=node_x,
            edge_index=self.pyg_data_template.edge_index,
            edge_attr=self.pyg_data_template.edge_attr,
            y=full_edge_labels,
        )

        return (
            pyg_data,
            observed_edge_indices,
        )  # Return actual observed indices for loss

    def train(self, historical_events: deque, epochs=10):
        pyg_data, observed_edge_indices = self.prepare_pyg_data_from_historical(
            historical_events
        )
        if not observed_edge_indices:
            logger.warning("No recent historical data to train on.")
            return

        self.model.train()
        for epoch in range(epochs):
            self.model_optimizer.zero_grad()
            out = self.model(pyg_data)

            # Only calculate loss for edges that had actual observed values
            loss = self.model_criterion(
                out[observed_edge_indices], pyg_data.y[observed_edge_indices]
            )
            loss.backward()
            self.model_optimizer.step()
            logger.debug(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        self._save_model()
        logger.info(f"Traffic predictor trained. Final loss: {loss.item():.4f}")

    def predict_traffic(
        self, current_historical_events: deque
    ) -> Dict[Tuple[int, int, int], float]:
        self.model.eval()
        with torch.no_grad():
            pyg_data, _ = self.prepare_pyg_data_from_historical(
                current_historical_events
            )
            predictions = self.model(pyg_data)

        predicted_travel_times = {}
        for i, (u, v, k) in enumerate(self.edge_list):
            predicted_travel_times[(u, v, k)] = max(
                1.0, predictions[i].item()
            )  # Ensure non-negative travel time

        logger.debug(f"Generated {len(predicted_travel_times)} traffic predictions.")
        return predicted_travel_times


if __name__ == "__main__":
    # Ensure dummy config and graph path
    config_file = "conf/environments/dev.yaml"
    graph_path_for_demo = "data_nexus/road_network_graph/preprocessed_tehran_graph.gml"  # Will be created if not exists
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(graph_path_for_demo), exist_ok=True)

    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    traffic_prediction:
      enabled: true
      model_dir: ml_ops/traffic_prediction_models
      learning_rate: 0.005
      gnn_model_params:
        node_features_dim: 5
        edge_features_dim: 2
        hidden_dim: 64
        num_layers: 2
      history_window_minutes: 60
"""
            )

    # Instantiate the engine
    predictor_engine = TrafficPredictionEngine(
        config_file, graph_path=graph_path_for_demo
    )

    # Simulate historical traffic events
    mock_traffic_stream = MockTrafficStream(predictor_engine.graph, history_size=100)
    print("Generating 100 historical traffic events...")
    for _ in range(100):
        mock_traffic_stream.generate_event()
    historical_events_deque = mock_traffic_stream.history

    # Train the GNN predictor
    print("\nTraining GNN traffic predictor...")
    predictor_engine.train(historical_events_deque, epochs=5)

    # Predict future traffic
    print("\nPredicting traffic for the current state...")
    predicted_times = predictor_engine.predict_traffic(historical_events_deque)

    print(f"\nExample predicted travel times for {len(predicted_times)} edges:")
    for i, ((u, v, k), time_val) in enumerate(predicted_times.items()):
        if i >= 5:
            break
        original_time = predictor_engine.graph[u][v][k].get("travel_time", 0.0)
        print(
            f"  Edge ({u}, {v}, {k}): Predicted={time_val:.2f}s (Base={original_time:.2f}s)"
        )

    print("\nGNNTrafficPredictor demo complete.")
