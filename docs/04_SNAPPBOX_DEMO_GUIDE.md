# SnappBox Demo Guide: Smart Logistics Optimization

This guide provides instructions for setting up and running a demonstration of the Smart Logistics & Delivery Optimization system, tailored for a SnappBox-like scenario. It showcases real-time routing, fleet management, and the impact of RL-driven decisions.

## Prerequisites

*   Docker and Docker Compose installed.
*   Python 3.9+ with `pip` and `venv`.
*   Git for cloning the repository.
*   At least 16GB RAM for local setup (for Kafka, Spark, DB, routing engines).

## Setup Instructions

### 1. Clone Repository and Environment Setup

```bash
git clone https://github.com/khoj-inc/logistics-optimization.git
cd logistics-optimization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2. Configure Environment
Edit conf/environments/dev.yaml to ensure all local service endpoints and paths are correct.

Ensure kafka: bootstrap_servers points to your local Kafka broker.
Ensure routing_engine: api_endpoint points to your local GraphHopper/Valhalla.
3. Prepare Geospatial Data
Download an OSM PBF file for Tehran (e.g., from Geofabrik). Place it in data_nexus/raw_osm_data/tehran_iran.osm.pbf.

Then, preprocess the graph:

python src/graph_routing_engine/osmnx_processor.py --config conf/osm_processing_config.yaml
This will generate data_nexus/road_network_graph/preprocessed_tehran_graph.gml.

4. Deploy Core Infrastructure (Docker Compose)
The deployment_ops/docker-compose.yaml (assuming such a file exists or will be created) will spin up Kafka, Zookeeper, PostgreSQL, Redis, GraphHopper/Valhalla.

docker-compose -f deployment_ops/docker-compose.yaml up -d
Wait for all services to be healthy.

5. Generate Simulation Scenario
Create initial drivers, orders, and traffic data for the demo:

python data_nexus/simulation_scenarios/generate_tehran_fleet_env.py
This will create a tehran_fleet_scenario.pkl in data_nexus/simulation_scenarios/.

6. Train/Load RL Agent & GNN
For demo purposes, you can either quickly train a simple agent or load a pre-trained one.

Option A: Load pre-trained: Ensure rl_model_registry/agent_policy_versions.json points to a valid model.
Option B: Quick Train:
python src/model_training/train_graph_gcn.py --config conf/rl_agent_params.yaml
python src/model_training/train_rl_agent.py --config conf/rl_agent_params.yaml --env-scenario data_nexus/simulation_scenarios/tehran_fleet_scenario.pkl
7. Start Services
Start Triton/TorchServe (for ML inference):

# Assuming Triton server is running via docker-compose or separately
# Example: docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repo:/models nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver --model-repository=/models
# Then copy your model to model_repo
Or if using TorchServe:

torchserve --start --model-store model_store --models rl_agent=path/to/your/rl_agent.mar
Start FastAPI Optimization API:

uvicorn src.deployment_core.optimization_api:app --host 0.0.0.0 --port 8000 --reload
Start Spark Kafka Consumer (for real-time data processing):

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 src/stream_processing/spark_kafka_consumer.py
8. Run the Demo Notebook
Open notebooks/snappbox_optimization_demo.ipynb and execute the cells. This notebook will:

Load the generated scenario.
Simulate order generation and vehicle movements.
Call the FastAPI optimization API for routing and fleet decisions.
Visualize the optimized routes and delivery progress on a map.
Showcase the impact of RL agent decisions on efficiency metrics.
9. Monitoring with Grafana
Access Grafana (usually http://localhost:3000) and import the provided dashboard (experiment_lab/monitoring/optimization_metrics_dashboard.json). This will visualize real-time KPIs from Prometheus.

Key Demo Scenarios to Showcase
Dynamic Re-routing: Introduce a sudden traffic jam and observe how the system recalculates optimal paths.
Order Batching: Demonstrate how the RL agent groups nearby orders for a single driver to maximize efficiency.
Capacity Management: Show how the system assigns orders based on vehicle capacity and current load.
ETA Accuracy: Compare predicted ETAs with actual delivery times in the simulation.

***
