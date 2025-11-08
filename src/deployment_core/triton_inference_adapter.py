# Triton client imports are handled conditionally to avoid import errors
try:
    import tritonclient.http as http_client
    TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    http_client = None
    TRITON_AVAILABLE = False

import logging
import numpy as np
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonInferenceAdapter:
    def __init__(self, config_path, environment="dev"):
        if not TRITON_AVAILABLE:
            raise ImportError("Triton client HTTP support not available. Install tritonclient with 'pip install tritonclient[http]'")

        self.config = self._load_config(config_path)
        self.env_config = self.config["environments"][environment]
        self.triton_url = (
            self.env_config["rl_agent"]["inference_endpoint"]
            .split("//")[1]
            .split("/predict")[0]
        )  # Extract host:port
        self.model_metadata = {}
        self.model_configs = {}

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    async def _get_model_metadata(self, model_name):
        if model_name not in self.model_metadata:
            try:
                self.model_metadata[model_name] = await http_client.get_model_metadata(
                    url=self.triton_url, model_name=model_name
                )
                self.model_configs[model_name] = await http_client.get_model_config(
                    url=self.triton_url, model_name=model_name
                )
            except Exception as e:
                logger.error(f"Failed to get metadata for model {model_name}: {e}")
                return None, None
        return self.model_metadata[model_name], self.model_configs[model_name]

    async def infer(self, model_name, input_data: dict, model_version: str = "1"):
        metadata, config = await self._get_model_metadata(model_name)
        if metadata is None or config is None:
            raise RuntimeError(f"Model {model_name} metadata/config not found.")

        inputs = []
        outputs = []

        for input_meta in metadata["inputs"]:
            input_name = input_meta["name"]
            if input_name not in input_data:
                raise ValueError(
                    f"Missing input '{input_name}' for model '{model_name}'."
                )

            data = input_data[input_name]
            # Ensure data is numpy array and correct type
            if not isinstance(data, np.ndarray):
                data = np.array(
                    data, dtype=http_client.str_to_np_dtype(input_meta["datatype"])
                )

            # Reshape if necessary (e.g., add batch dimension)
            if len(data.shape) < len(input_meta["shape"]):
                data = np.expand_dims(data, axis=0)

            inputs.append(
                http_client.InferInput(
                    input_name, list(data.shape), input_meta["datatype"]
                )
            )
            inputs[-1].set_contents_as_numpy(data)

        for output_meta in metadata["outputs"]:
            outputs.append(
                http_client.InferRequestedOutput(output_meta["name"], binary_data=True)
            )

        try:
            response = await http_client.infer(
                url=self.triton_url,
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                model_version=model_version,
            )

            results = {}
            for output_meta in metadata["outputs"]:
                output_name = output_meta["name"]
                results[output_name] = response.as_numpy(output_name)

            return results
        except Exception as e:
            logger.error(f"Triton inference failed for model {model_name}: {e}")
            raise


if __name__ == "__main__":
    # For local demonstration, you need a running Triton server
    # Example: docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repo:/models nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver --model-repository=/models
    # And a dummy model in model_repo, e.g., a PyTorch model exported for Triton.

    # Dummy model_repo structure for example:
    # model_repo/
    # ├── rl_agent_model/
    # │   └── 1/
    # │       └── model.pt (PyTorch traced model)
    # └── gnn_embedder/
    #     └── 1/
    #         └── model.pt

    # Ensure dummy config exists for startup
    conf_dir = "conf/environments"
    os.makedirs(conf_dir, exist_ok=True)
    dev_config_path = os.path.join(conf_dir, "dev.yaml")
    if not os.path.exists(dev_config_path):
        with open(dev_config_path, "w") as f:
            f.write(
                """
environment: development
redis:
  host: localhost
  port: 6379
  db: 0
rl_agent:
  inference_endpoint: http://localhost:8001/v2/models/rl_agent_model/infer
  model_version: dev_v1.0
routing_engine:
  api_endpoint: http://localhost:8080/route
"""
            )

    adapter = TritonInferenceAdapter(dev_config_path)

    async def test_inference():
        logger.info("Testing RL Agent model inference...")
        try:
            # Assume rl_agent_model expects 'input_obs' as a float32 array
            dummy_obs = np.random.rand(1, 128).astype(np.float32)
            results = await adapter.infer("rl_agent_model", {"input_obs": dummy_obs})
            logger.info(f"RL Agent Model Inference Result Keys: {results.keys()}")
            for key, val in results.items():
                logger.info(f"  {key} shape: {val.shape}, sample: {val[0, :5]}")
        except Exception as e:
            logger.error(f"RL Agent inference test failed: {e}")

        logger.info("Testing GNN Embedder model inference...")
        try:
            # Assume gnn_embedder expects 'node_features' and 'edge_index'
            dummy_node_features = np.random.rand(10, 4).astype(np.float32)
            dummy_edge_index = np.array(
                [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64
            )  # Example edges

            # Triton client for TorchGeometric models might need careful input shaping
            # For simplicity, if GNN is traced as single input 'graph_data' combining nodes and edges or two separate inputs

            # If exported as two separate inputs:
            results_gnn = await adapter.infer(
                "gnn_embedder",
                {"node_features": dummy_node_features, "edge_index": dummy_edge_index},
                model_version="1",
            )
            logger.info(
                f"GNN Embedder Model Inference Result Keys: {results_gnn.keys()}"
            )
            for key, val in results_gnn.items():
                logger.info(f"  {key} shape: {val.shape}, sample: {val[0, :5]}")

        except Exception as e:
            logger.error(f"GNN Embedder inference test failed: {e}")

    import asyncio

    asyncio.run(test_inference())
