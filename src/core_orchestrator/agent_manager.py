import numpy as np
import yaml
import os
import logging
import json
from typing import List, Any

try:
    from ..learning.multi_agent_rl_policy import MultiAgentPolicyContainer
except ImportError:
    MultiAgentPolicyContainer = None
try:
    from ..deployment_core.triton_inference_adapter import TritonInferenceAdapter
except (ImportError, RuntimeError):
    TritonInferenceAdapter = None
from ..feature_forge.graph_embedding_features import GraphEmbeddingFeatures
from ..feature_forge.feature_store_client import FeatureStoreClient
from ..routing.osmnx_processor import OSMNxProcessor  # For graph context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.env_config = self.config["environments"][environment]
        self.rl_agent_config = self.env_config["rl_agent"]
        self.deployment_environment = environment

        # Initialize Triton Adapter for model inference (if using Triton)
        try:
            self.triton_adapter = TritonInferenceAdapter(config_path, environment)
        except Exception as e:
            logger.warning(f"Triton adapter initialization failed: {e}. Triton features will be disabled.")
            self.triton_adapter = None

        # Initialize Feature Store Client to retrieve real-time features
        self.feature_store_client = FeatureStoreClient(config_path, environment)

        # Initialize GNN Embedder for graph context
        self.osm_processor = OSMNxProcessor(
            self.env_config.get(
                "osm_processing_config_path", "conf/osm_processing_config.yaml"
            )
        )
        self.current_graph = (
            self.osm_processor.get_graph()
        )  # Load once, update dynamically
        self.graph_embedder = GraphEmbeddingFeatures(
            config_path=self.env_config.get(
                "rl_agent_params_config_path", "conf/rl_agent_params.yaml"
            ),
            graph_path=self.osm_processor.output_path,
        )
        # Load GNN model weights
        gnn_model_path = os.environ.get(
            "GNN_MODEL_PATH", self.rl_agent_config.get("gnn_model_path")
        )
        if gnn_model_path and os.path.exists(gnn_model_path):
            self.graph_embedder.load_model_weights(gnn_model_path)
        else:
            logger.warning(
                f"GNN model not found at {gnn_model_path}. Embeddings will be random."
            )

        # RL policy container (for abstracting policies, might not be used if directly calling Triton)
        # However, it helps define observation/action spaces and policy mapping
        if MultiAgentPolicyContainer is not None:
            self.multi_agent_policy_container = MultiAgentPolicyContainer(
                self.config["environments"][environment].get(
                    "rl_agent_params_config_path", "conf/rl_agent_params.yaml"
                )
            )
        else:
            self.multi_agent_policy_container = None
        logger.info("AgentManager initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    kafka:
      bootstrap_servers: ['localhost:9092']
      topic_traffic_data: dev_traffic_events
      topic_order_data: dev_order_events
      topic_telemetry_data: dev_telemetry_stream
    rl_agent:
      inference_endpoint: http://localhost:8001/v2/models/rl_agent_model/infer
      model_version: dev_v1.0
      explore_probability: 0.05
      gnn_model_path: rl_model_registry/gcn_model.pth
      rl_checkpoint_path: rl_model_registry/dev_v1.0/checkpoint_000100
    osm_processing_config_path: conf/osm_processing_config.yaml
    rl_agent_params_config_path: conf/rl_agent_params.yaml
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    async def get_agent_action(
        self, agent_id: str, current_state: dict[str, Any], available_actions: List[str]
    ) -> str:
        """
        Retrieves the optimal action for a given agent by fetching features,
        generating graph embeddings, and calling the RL inference service.
        """
        # 1. Fetch real-time driver/fleet features from Feature Store
        driver_features = self.feature_store_client.get_feature("driver", agent_id)
        # Example of how current_state might integrate with fetched features
        # In a full system, you'd combine/process these more robustly.
        combined_state = {
            **current_state,
            **(driver_features if driver_features else {}),
        }

        # 2. Prepare GNN input features
        # Assuming current_graph is updated and node features are accessible for embedding
        pyg_data, node_map = self.graph_embedder.preprocess_graph_for_pyg(
            self.current_graph
        )
        node_embeddings_tensor = self.graph_embedder.generate_embeddings(
            pyg_data
        )  # Returns torch tensor
        node_embeddings_np = (
            node_embeddings_tensor.cpu().numpy()
        )  # Convert to numpy for Triton/RLlib

        # For a simple agent, map agent_id to a single node embedding for its location
        # This is a simplification; a full system would have more complex feature engineering
        agent_node_id = combined_state.get("current_node_id")
        if agent_node_id is not None and agent_node_id in node_map:
            agent_node_pyg_idx = node_map[agent_node_id]
            relevant_graph_features = node_embeddings_np[agent_node_pyg_idx].flatten()
        else:
            logger.warning(
                f"Agent {agent_id} has no valid node_id or node_id not in graph. Using default graph features."
            )
            relevant_graph_features = (
                node_embeddings_np[0].flatten()
                if len(node_embeddings_np) > 0
                else np.zeros(self.graph_embedder.output_dim)
            )

        # 3. Create observation for the RL model
        # The exact observation structure depends on the RL model's input expectations.
        # This is a placeholder for careful feature engineering.
        # Example: [driver_speed, driver_load, driver_status, graph_embedding_vector...]

        # Simple example: combine some direct state elements with graph features
        obs_elements = [
            combined_state.get("speed_mps", 0.0),
            combined_state.get("current_load", 0.0),
            1.0 if combined_state.get("status") == "available" else 0.0,
        ]

        # Pad obs_elements to a consistent size if they vary, before concatenating
        # For simplicity, ensure graph features are consistent size

        # In a real system, the observation would be constructed carefully based on the RL agent's input spec
        # Here, we use a simple concatenation of a few numeric driver features and the graph embedding

        # Construct a dummy observation for Triton, matching a potential input shape (e.g., 128)
        # This is highly specific to the actual RL model's input signature.
        if self.multi_agent_policy_container is not None:
            dummy_observation = np.random.rand(
                self.multi_agent_policy_container.get_policy_obs_space_shape(
                    "driver_policy"
                )[0]
            ).astype(np.float32)
        else:
            dummy_observation = np.random.rand(128).astype(np.float32)  # Default size
        # Override with actual features if available, e.g., first few elements
        if relevant_graph_features.size > 0:
            dummy_observation[
                : min(dummy_observation.size, relevant_graph_features.size)
            ] = relevant_graph_features[
                : min(dummy_observation.size, relevant_graph_features.size)
            ]

        # Pass through Triton for inference
        model_name = "rl_agent_model"  # Configured in Triton
        model_version = self.rl_agent_config["model_version"]

        try:
            inference_results = await self.triton_adapter.infer(
                model_name=model_name,
                input_data={
                    "input_obs": dummy_observation.reshape(1, -1)
                },  # Triton expects batch dimension
                model_version=model_version,
            )

            # Extract action from inference results. Output format depends on the RL model.
            # Assuming it outputs 'action' or 'logits' that need to be processed.
            # Example: if output is logits, apply argmax.
            if "action" in inference_results:
                raw_action_output = inference_results["action"].flatten()
            elif "output_logits" in inference_results:  # Common for discrete actions
                raw_action_output = np.argmax(
                    inference_results["output_logits"], axis=-1
                ).flatten()
            else:
                logger.error(
                    f"RL model {model_name} output key not recognized: {inference_results.keys()}"
                )
                return "wait"  # Default fallback

            # Map raw_action_output (e.g., an integer) to one of the available_actions
            if len(available_actions) > 0:
                chosen_action_idx = raw_action_output[0] % len(available_actions)
                chosen_action = available_actions[chosen_action_idx]
            else:
                chosen_action = "wait"  # Fallback if no actions are available

            logger.info(f"Agent {agent_id} chose action: {chosen_action}")
            return chosen_action

        except Exception as e:
            logger.error(f"Error during RL agent inference for {agent_id}: {e}")
            return "wait"  # Fallback action

    async def update_graph_context(self, updated_graph: any):
        """
        Updates the internal graph used by the GNN embedder.
        This would typically be called when dynamic traffic data or road network changes are significant.
        """
        self.current_graph = updated_graph
        # Re-initialize or update the GNN embedder with the new graph if its structure changed significantly
        # For performance, only re-process if node/edge structure changes, not just attributes.
        # For simplicity, we assume we can regenerate embeddings on the fly.
        self.graph_embedder.graph = updated_graph
        logger.info("AgentManager graph context updated.")


if __name__ == "__main__":
    # Ensure necessary config files exist for startup
    conf_dir = "conf/environments"
    os.makedirs(conf_dir, exist_ok=True)
    dev_config_path = os.path.join(conf_dir, "dev.yaml")

    # Create dummy environment config
    if not os.path.exists(dev_config_path):
        with open(dev_config_path, "w") as f:
            f.write(
                """
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    kafka:
      bootstrap_servers: ['localhost:9092']
      topic_traffic_data: dev_traffic_events
      topic_order_data: dev_order_events
      topic_telemetry_data: dev_telemetry_stream
    rl_agent:
      inference_endpoint: http://localhost:8001/v2/models/rl_agent_model/infer
      model_version: dev_v1.0
      explore_probability: 0.05
      gnn_model_path: rl_model_registry/gcn_model.pth
      rl_checkpoint_path: rl_model_registry/dev_v1.0/checkpoint_000100
    osm_processing_config_path: conf/osm_processing_config.yaml
    rl_agent_params_config_path: conf/rl_agent_params.yaml
"""
            )

    # Create dummy OSM processing config
    osm_config_path = "conf/osm_processing_config.yaml"
    os.makedirs(os.path.dirname(osm_config_path), exist_ok=True)
    if not os.path.exists(osm_config_path):
        with open(osm_config_path, "w") as f:
            f.write(
                """
osm_data:
  pbf_path: data_nexus/raw_osm_data/tehran_iran.osm.pbf
  bounding_box: [35.5, 51.2, 35.9, 51.7]
  cache_dir: data_nexus/road_network_graph/
graph_serialization:
  output_format: gml
  output_path: data_nexus/road_network_graph/preprocessed_tehran_graph.gml
osmnx:
  graph_type: drive
  network_type: drive_service
preprocessing_steps: []
"""
            )

    # Create dummy RL agent params config
    rl_params_config_path = "conf/rl_agent_params.yaml"
    os.makedirs(os.path.dirname(rl_params_config_path), exist_ok=True)
    if not os.path.exists(rl_params_config_path):
        with open(rl_params_config_path, "w") as f:
            f.write(
                """
algorithm: PPO
multi_agent_config:
  policies:
    driver_policy:
      obs_space: [128]
      action_space: [5]
    fleet_manager_policy:
      obs_space: [256]
      action_space: [10]
gnn_embedding:
  input_dim: 10 # Example feature dimension for nodes
  hidden_dim: 64
  output_dim: 32 # Output embedding dimension
  num_layers: 2
"""
            )

    # Create dummy model directories and files for GNN and RL
    os.makedirs("rl_model_registry", exist_ok=True)
    with open("rl_model_registry/gcn_model.pth", "w") as f:
        f.write("dummy gnn model")
    os.makedirs("rl_model_registry/dev_v1.0", exist_ok=True)
    with open("rl_model_registry/dev_v1.0/checkpoint_000100", "w") as f:
        f.write("dummy rl checkpoint")

    # This example requires a running Triton server if `triton_inference_adapter` is to connect.
    # For a standalone demo without a running Triton, the TritonInferenceAdapter itself needs to be mocked.
    # The current `triton_inference_adapter.py` main block shows how to run a dummy Triton.
    # For this `AgentManager` test, we'll assume Triton is up or the adapter will gracefully fallback (which it does via logging errors).

    # Initialize Redis for FeatureStoreClient (docker run --name some-redis -p 6379:6379 -d redis)
    import redis

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0)
        r.ping()
        print("Connected to Redis.")
        # Populate dummy driver features
        r.set(
            "driver_feature:driver_A",
            json.dumps(
                {
                    "speed_mps": 10.0,
                    "current_load": 5.0,
                    "status": "available",
                    "current_node_id": 1,
                }
            ),
        )
        r.set(
            "driver_feature:driver_B",
            json.dumps(
                {
                    "speed_mps": 12.0,
                    "current_load": 2.0,
                    "status": "on_route",
                    "current_node_id": 5,
                }
            ),
        )
    except redis.exceptions.ConnectionError:
        print("Redis not running. Feature store client might fail.")

    import asyncio

    async def demo_agent_manager():
        manager = AgentManager(dev_config_path)

        # Simulate a driver's current state and available actions
        driver_state_A = {
            "current_node_id": 1,
            "status": "available",
            "speed_mps": 10.0,
            "current_load": 5.0,
        }
        available_actions_A = ["move_to_node_2", "assign_order_XYZ", "wait"]

        print(
            f"\nRequesting action for driver_A: {driver_state_A}, Actions: {available_actions_A}"
        )
        action_A = await manager.get_agent_action(
            "driver_A", driver_state_A, available_actions_A
        )
        print(f"Driver_A's chosen action: {action_A}")

        driver_state_B = {
            "current_node_id": 5,
            "status": "on_route",
            "speed_mps": 12.0,
            "current_load": 2.0,
        }
        available_actions_B = ["continue_route", "report_issue"]
        print(
            f"\nRequesting action for driver_B: {driver_state_B}, Actions: {available_actions_B}"
        )
        action_B = await manager.get_agent_action(
            "driver_B", driver_state_B, available_actions_B
        )
        print(f"Driver_B's chosen action: {action_B}")

        # Simulate graph update
        # For a real graph, you'd load a new GML or modify in memory
        new_graph = manager.current_graph.copy()
        if new_graph.number_of_nodes() > 1:
            first_node = list(new_graph.nodes())[0]
            second_node = list(new_graph.nodes())[1]
            if not new_graph.has_edge(first_node, second_node):
                new_graph.add_edge(
                    first_node,
                    second_node,
                    key=0,
                    travel_time=100,
                    length=500,
                    traffic_factor=1.0,
                )
                print(
                    f"\nSimulating graph update: Added edge ({first_node}, {second_node})."
                )
                await manager.update_graph_context(new_graph)
            else:
                print(
                    "\nSimulating graph update: No new edges added, modifying existing for demo."
                )
                new_graph[first_node][second_node][0][
                    "traffic_factor"
                ] = 1.5  # Simulate traffic change
                await manager.update_graph_context(new_graph)
        else:
            print("\nCannot simulate graph update with too few nodes.")

        print("\nAgent Manager demo complete.")

    asyncio.run(demo_agent_manager())
