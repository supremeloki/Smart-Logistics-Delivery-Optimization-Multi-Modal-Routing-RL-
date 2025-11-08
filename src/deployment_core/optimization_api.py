from fastapi import FastAPI, HTTPException, status
import logging
import yaml
import os

from routing.graphhopper_api_client import GraphHopperAPIClient
from routing.astar_optimization_logic import AStarRouting
from learning.multi_agent_rl_policy import MultiAgentPolicyContainer
from feature_forge.real_time_traffic_features import RealTimeTrafficFeatures
from feature_forge.batch_order_features import BatchOrderFeatures
from feature_forge.graph_embedding_features import (
    GraphEmbeddingFeatures,
)
from pydantic import BaseModel
from typing import List, Dict, Optional
import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic Models ---
class Coordinates(BaseModel):
    lat: float
    lon: float


class RoutingRequest(BaseModel):
    origin: Coordinates
    destination: Coordinates
    vehicle_profile: str = "car"
    avoid_segments: Optional[List[Dict]] = None  # e.g., [{"u": 1, "v": 2}]
    current_time_utc: str  # ISO format
    vehicle_properties: Optional[Dict] = None


class RouteResponse(BaseModel):
    distance_meters: float
    time_seconds: float
    path_points: List[Coordinates]
    instructions: List[Dict] = []


class OrderRequest(BaseModel):
    order_id: str
    pickup_node_id: int  # OSMnx node ID
    delivery_node_id: int  # OSMnx node ID
    weight: float
    volume: float
    pickup_time_start: str  # ISO format
    pickup_time_end: str  # ISO format
    delivery_time_latest: str  # ISO format


class DriverStatus(BaseModel):
    driver_id: str
    current_node_id: int
    vehicle_type: str
    current_load: float
    capacity: float
    status: str  # 'available', 'on_route'


class OptimizationRequest(BaseModel):
    current_drivers: List[DriverStatus]
    pending_orders: List[OrderRequest]
    current_time_utc: str


class OptimizationResponse(BaseModel):
    assignments: List[Dict] = (
        []
    )  # e.g., [{"driver_id": "d1", "order_ids": ["o1"], "route": [...]}]
    re_routes: List[Dict] = []  # e.g., [{"driver_id": "d2", "new_route": [...]}]


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Smart Logistics Optimization API",
    description="API for multi-modal routing and RL-driven fleet optimization.",
    version="1.0.0",
)


# --- Dependency Loading ---
@app.on_event("startup")
async def startup_event():
    global gh_client, astar_router, rl_agent_inferer, traffic_features, order_features, graph_embeddings, current_graph

    # Load configuration
    env_config_path = os.getenv("ENV_CONFIG_PATH", "conf/environments/dev.yaml")
    routing_config_path = "conf/routing_engine_config.yaml"
    rl_agent_config_path = "conf/rl_agent_params.yaml"
    osm_processing_config_path = "conf/osm_processing_config.yaml"

    with open(env_config_path, "r") as f:
        yaml.safe_load(f)
    with open(routing_config_path, "r") as f:
        yaml.safe_load(f)
    with open(rl_agent_config_path, "r") as f:
        rl_config = yaml.safe_load(f)
    with open(osm_processing_config_path, "r") as f:
        osm_config = yaml.safe_load(f)

    # Load Graph (preprocessed)
    graph_path = osm_config["graph_serialization"]["output_path"]
    if not os.path.exists(graph_path):
        logger.error(
            f"Preprocessed graph not found at {graph_path}. Run osmnx_processor.py first."
        )
        raise RuntimeError("Graph not found")
    current_graph = nx.read_gml(graph_path)

    # Initialize Routing Clients
    gh_client = GraphHopperAPIClient(routing_config_path)
    astar_router = AStarRouting(current_graph, routing_config_path)

    # Initialize Feature Extractors
    traffic_features = RealTimeTrafficFeatures(env_config_path)
    order_features = BatchOrderFeatures(env_config_path)

    # Initialize Graph Embedding
    gnn_model_path = os.getenv("GNN_MODEL_PATH", "rl_model_registry/gcn_model.pth")
    # For now, GraphEmbeddingFeatures needs config, so we pass one of them
    graph_embeddings = GraphEmbeddingFeatures(rl_agent_config_path, graph_path)
    if os.path.exists(gnn_model_path):
        graph_embeddings.load_model_weights(gnn_model_path)
    else:
        logger.warning(
            f"GNN model not found at {gnn_model_path}. Embeddings will be random."
        )

    # Initialize RL Agent Inferer
    rl_agent_inferer = MultiAgentPolicyContainer(rl_config)
    rl_checkpoint_path = os.getenv(
        "RL_CHECKPOINT_PATH", "rl_model_registry/latest_checkpoint"
    )
    if os.path.exists(rl_checkpoint_path):
        rl_agent_inferer.load_checkpoint(rl_checkpoint_path)
        logger.info(f"RL Agent loaded from {rl_checkpoint_path}")
    else:
        logger.warning(
            "RL Agent checkpoint not found. Agent will use default random policy or fail."
        )


# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Logistics Optimization API is running"}


# --- Routing Endpoint ---
@app.post("/route", response_model=RouteResponse)
async def get_optimized_route(request: RoutingRequest):
    # Retrieve dynamic traffic factor
    # This would involve looking up traffic for segments on a potential path
    # For simplicity, we'll assume the routing engines handle it internally or A* uses pre-updated graph

    # Use GraphHopper for external routing
    origin_coords = (request.origin.lat, request.origin.lon)
    destination_coords = (request.destination.lat, request.destination.lon)

    gh_route = gh_client.get_route(
        origin_coords,
        destination_coords,
        request.vehicle_profile,
        avoid_segments=request.avoid_segments,
    )

    if gh_route:
        # Convert raw coordinates to Pydantic Coordinates
        path_coords = [Coordinates(lat=p[1], lon=p[0]) for p in gh_route["points"]]
        return RouteResponse(
            distance_meters=gh_route["distance"],
            time_seconds=gh_route["time_seconds"],
            path_points=path_coords,
            instructions=gh_route["instructions"],
        )

    # Fallback to A* if GraphHopper fails or for custom logic
    # Need to map lat/lon to graph nodes for A*
    origin_node = ox.nearest_nodes(
        current_graph, request.origin.lon, request.origin.lat
    )
    destination_node = ox.nearest_nodes(
        current_graph, request.destination.lon, request.destination.lat
    )

    path_nodes, cost = astar_router.find_path(
        origin_node,
        destination_node,
        current_time=request.current_time_utc,
        vehicle_properties=request.vehicle_properties,
    )
    if path_nodes:
        # Reconstruct path points from nodes
        path_points_astar = []
        for node_id in path_nodes:
            node_data = current_graph.nodes[node_id]
            path_points_astar.append(
                Coordinates(lat=node_data["y"], lon=node_data["x"])
            )

        # Estimate distance and time from A* path (simplified)
        distance_astar = sum(
            current_graph[u][v][0].get("length", 0)
            for u, v in zip(path_nodes[:-1], path_nodes[1:])
        )
        time_astar = sum(
            current_graph[u][v][0].get("travel_time", 0)
            * current_graph[u][v][0].get("traffic_factor", 1.0)
            for u, v in zip(path_nodes[:-1], path_nodes[1:])
        )

        return RouteResponse(
            distance_meters=distance_astar,
            time_seconds=time_astar,
            path_points=path_points_astar,
            instructions=[],  # A* typically doesn't generate turn-by-turn
        )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No route found.")


# --- Fleet Optimization Endpoint (RL-driven) ---
@app.post("/optimize_fleet", response_model=OptimizationResponse)
async def optimize_fleet(request: OptimizationRequest):
    # Prepare observations for the Multi-Agent RL Policy
    # This involves combining various features:
    # 1. Driver states (location, load, status)
    # 2. Order states (pickup/delivery nodes, time windows, weight/volume)
    # 3. Real-time traffic features (grid, edge-specific)
    # 4. Graph embeddings (from GNN)

    # Simplified feature aggregation for observation
    pass

    # Get graph embeddings
    pyg_data, node_map = graph_embeddings.preprocess_graph_for_pyg(current_graph)
    embeddings = graph_embeddings.generate_embeddings(pyg_data)

    # Traffic grid features
    traffic_grid_features = traffic_features.generate_traffic_grid_features(
        current_graph, request.current_time_utc
    )

    # Order density features
    pending_orders_df = pd.DataFrame([o.dict() for o in request.pending_orders])
    order_density_features = order_features.generate_order_density_features(
        pending_orders_df, current_graph
    )

    # Construct observations for RL agents
    agent_observations = {}

    # For dispatcher (fleet manager)
    # Combine global state: traffic grid, order density, aggregate driver/order stats, graph embeddings
    dispatcher_global_features = np.concatenate(
        [
            traffic_grid_features.flatten(),
            order_density_features.flatten(),
            np.array([len(request.current_drivers), len(request.pending_orders)]),
            embeddings.cpu()
            .numpy()
            .flatten()[:100],  # Partial embeddings for global context
        ]
    )
    dispatcher_obs = np.pad(
        dispatcher_global_features,
        (0, 256 - len(dispatcher_global_features)),
        "constant",
    )  # Pad to fixed size
    agent_observations["dispatcher_0"] = dispatcher_obs[
        :256
    ]  # Ensure it matches expected obs_space

    # For individual drivers (if decentralized actions are considered)
    for driver_status in request.current_drivers:
        node_embedding = graph_embeddings.get_node_embedding(
            driver_status.current_node_id, embeddings, node_map
        )
        driver_local_features = np.concatenate(
            [
                node_embedding,
                np.array(
                    [
                        driver_status.current_load,
                        driver_status.capacity,
                        1 if driver_status.status == "available" else 0,
                    ]
                ),
            ]
        )
        driver_obs = np.pad(
            driver_local_features, (0, 128 - len(driver_local_features)), "constant"
        )  # Pad to fixed size
        agent_observations[f"driver_{driver_status.driver_id}"] = driver_obs[
            :128
        ]  # Ensure it matches expected obs_space

    # Infer actions from the RL agent
    try:
        actions = rl_agent_inferer.infer_actions(agent_observations)
    except Exception as e:
        logger.error(f"Error during RL agent inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RL agent inference failed.",
        )

    # Parse actions into assignments and re-routes
    assignments = []
    re_routes = []

    # Example: If dispatcher action is a list of (driver_id, order_id) assignments
    # This logic needs to match the RL agent's action space definition.
    # For demo, let's assume dispatcher output is directly actionable.
    if "dispatcher_0" in actions:
        dispatcher_action = actions["dispatcher_0"]
        # Example: dispatcher_action might be an index to a predefined assignment strategy
        # or a complex structure. Here, we mock a simple assignment.

        if request.pending_orders and request.current_drivers:
            chosen_driver = request.current_drivers[
                dispatcher_action % len(request.current_drivers)
            ]
            chosen_order = request.pending_orders[0]  # Pick first pending
            assignments.append(
                {
                    "driver_id": chosen_driver.driver_id,
                    "order_ids": [chosen_order.order_id],
                    "target_node": chosen_order.pickup_node_id,  # This should come from RL policy if complex
                }
            )
            logger.info(
                f"Assigned order {chosen_order.order_id} to {chosen_driver.driver_id} (via dispatcher)."
            )

    # Example: If driver actions are re-routing decisions or path segment choices
    for driver_status in request.current_drivers:
        driver_id_key = f"driver_{driver_status.driver_id}"
        if driver_id_key in actions:
            driver_action = actions[driver_id_key]
            # Mock re-route: driver wants to go to a nearby node
            if (
                driver_action == 0
            ):  # Example action: request re-route to closest available task
                if driver_status.status == "on_route" and driver_status.current_node_id:
                    # Find a new target (e.g., closest node from pending orders)
                    if request.pending_orders:
                        closest_order_node = min(
                            [o.pickup_node_id for o in request.pending_orders],
                            key=lambda node_id: ox.distance.euclidean(
                                current_graph.nodes[driver_status.current_node_id]["x"],
                                current_graph.nodes[driver_status.current_node_id]["y"],
                                current_graph.nodes[node_id]["x"],
                                current_graph.nodes[node_id]["y"],
                            ),
                        )
                        # Recalculate route using A* or GraphHopper
                        new_path_nodes, _ = astar_router.find_path(
                            driver_status.current_node_id, closest_order_node
                        )
                        if new_path_nodes:
                            new_route_points = [
                                Coordinates(
                                    lat=current_graph.nodes[n]["y"],
                                    lon=current_graph.nodes[n]["x"],
                                )
                                for n in new_path_nodes
                            ]
                            re_routes.append(
                                {
                                    "driver_id": driver_status.driver_id,
                                    "new_route_points": new_route_points,
                                }
                            )
                            logger.info(
                                f"Driver {driver_status.driver_id} re-routed to {closest_order_node}."
                            )

    return OptimizationResponse(assignments=assignments, re_routes=re_routes)


# --- Example of Real-time Traffic Update (Kafka Consumer would push to Redis, but API could also be direct) ---
class TrafficUpdateRequest(BaseModel):
    edge_u: int
    edge_v: int
    edge_key: int = 0  # For multigraph
    current_travel_time: float
    timestamp_utc: str


@app.post("/update_traffic")
async def update_traffic_data(request: TrafficUpdateRequest):
    traffic_features.update_traffic_data(
        request.edge_u,
        request.edge_v,
        request.edge_key,
        request.current_travel_time,
        pd.to_datetime(request.timestamp_utc),
    )
    return {"message": "Traffic data updated successfully."}


if __name__ == "__main__":
    # For local development:
    # 1. Ensure env configs exist (conf/environments/dev.yaml, conf/routing_engine_config.yaml, etc.)
    # 2. Ensure a preprocessed graph exists (data_nexus/road_network_graph/preprocessed_tehran_graph.gml)
    #    Run `python src/graph_routing_engine/osmnx_processor.py`
    # 3. (Optional) Run `python src/model_training/train_graph_gcn.py` to get a GNN model
    # 4. (Optional) Run `python src/model_training/train_rl_agent.py` to get an RL checkpoint

    # Set environment variables for paths if not using default
    # os.environ["ENV_CONFIG_PATH"] = "conf/environments/dev.yaml"
    # os.environ["GNN_MODEL_PATH"] = "rl_model_registry/gcn_model.pth"
    # os.environ["RL_CHECKPOINT_PATH"] = "rl_model_registry/latest_checkpoint"

    # Simulate startup events for local testing
    import asyncio

    asyncio.run(startup_event())

    # Run Uvicorn directly for local development
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
