# Smart Logistics & Delivery Optimization (Multi-Modal Routing + RL)

This project implements an intelligent, multi-modal logistics optimization system using Deep Reinforcement Learning (DRL) and graph-based routing algorithms. The system optimizes delivery routes, fleet management, and real-time decision-making across urban environments, integrating AI-driven demand forecasting, dynamic pricing, and autonomous vehicle coordination.

## Key Features

- **Multi-Modal Routing**: Combines road, drone, and autonomous vehicle routing with RL optimization
- **Real-Time Adaptation**: Dynamic route re-planning based on traffic, weather, and driver conditions
- **Fleet Management**: EV energy optimization, driver wellbeing monitoring, and adaptive workforce allocation
- **Scalable Architecture**: Modular design supporting distributed deployment and edge computing
- **Data-Driven Insights**: Integrated feature store, simulation environments, and performance monitoring
- **Advanced RL Benchmarking**: Professional evaluation framework using lono_libs for comprehensive agent comparison

## Key Capabilities

The project consists of the following modules, each performing intelligent and optimization tasks:

*   **Dynamic Pricing Engine - `src/economic_optimization/dynamic_pricing_engine.py`**:
    *   Adjusts service pricing in real-time based on demand, supply, traffic, weather conditions, and order urgency.
    *   Calculates driver incentives based on fatigue and stress levels.
    *   Applies customer loyalty discounts.

*   **AV Control Interface - `src/autonomous_systems/av_control_interface.py`**:
    *   Sends navigation commands to autonomous vehicles (AVs).
    *   Receives and processes real-time telemetry streams from AVs.
    *   Monitors AV health status and critical alerts.

*   **Geospatial Demand Predictor - `src/demand_forecasting/geospatial_demand_predictor.py`**:
    *   Predicts future demand for various nodes in a city graph using a (mock) GNN model.
    *   Identifies high-demand hotspots.
    *   Considers temporal, spatial, Point-of-Interest (POI), and event factors.

*   **EV Energy Optimizer - `src/fleet_management/ev_energy_optimizer.py`**:
    *   Monitors battery levels of the electric vehicle fleet.
    *   Finds optimal charging stations based on location, cost, and charging speed.
    *   Plans charging routes for EVs with low or critical battery levels.
    *   Predicts energy consumption considering distance, elevation, and traffic.

*   **DRL Predictive Router - `src/graph_routing_engine/drl_predictive_router.py`**:
    *   Generates optimized routes for drivers and orders using Deep Reinforcement Learning (DRL).
    *   Considers factors such as travel time, fuel consumption, driver fatigue, and delivery priority.
    *   Dynamically re-evaluates routes based on real-time conditions (traffic, weather).

*   **Adaptive Workforce Optimizer - `src/worker_management/adaptive_workforce_optimizer.py`**:
    *   Assigns tasks to available workers based on skills, location, fatigue levels, and worker preferences.
    *   Includes penalty mechanisms for skill mismatches, preferences, and high fatigue.
    *   Runs a continuous optimization cycle for optimal task assignment.

*   **CV Quality Control System - `src/warehouse_operations/cv_quality_control_system.py`**:
    *   Inspects package quality at quality control points using (mock) computer vision models.
    *   Detects defects and verifies labels using OCR.
    *   Initiates automated actions like repackaging or diverting for anomalous items.

*   **Product Traceability Ledger - `src/supply_chain_resiliency/product_traceability_ledger.py`**:
    *   Records product events throughout the supply chain on a (mock) blockchain ledger.
    *   Provides capabilities to retrieve full traceability history and current product ownership.
    *   Enhances supply chain transparency and resilience.

*   **Smart Irrigation Optimizer - `src/monitoring/smart_irrigation_optimizer.py`**:
    *   Determines irrigation needs based on soil sensor data (moisture, temperature), weather forecasts, and plant profiles.
    *   Automatically opens and closes irrigation valves for optimal duration and flow rate.
    *   Optimizes water usage by accounting for forecasted precipitation.

*   **Public Safety Alert System - `src/smart_city/public_safety_alert_system.py`**:
    *   Monitors incident reports and real-time anomalies (e.g., from cameras).
    *   Initiates public alerts or police dispatch based on incident severity.
    *   Notifies citizens based on their location, alert radius, and preferred crime types.

*   **AI-Driven Diagnosis Assistant - `src/healthcare/ai_driven_diagnosis_assistant.py`**:
    *   Analyzes patient cases by integrating patient records, lab results, and (mock) imaging scans.
    *   Leverages medical imaging AI for scan interpretation and NLP for symptom extraction.
    *   Identifies potential conditions and provides diagnostic and treatment recommendations based on medical guidelines.

## Project Structure

```
.
├── conf/                           # Configuration files
│   ├── environments/               # Environment-specific configs (dev, prod)
│   ├── osm_processing_config.yaml  # OSM data processing settings
│   └── routing_engine_config.yaml  # Routing engine parameters
├── data_nexus/                     # Data ingestion and simulation
│   ├── raw_osm_data/               # Raw OpenStreetMap data
│   ├── road_network_graph/         # Processed graph data
│   └── simulation_scenarios/       # Fleet simulation environments
├── deployment_ops/                 # Deployment configurations
│   ├── docker/                     # Docker containers for services
│   └── kubernetes/                 # K8s manifests
├── docs/                           # Documentation
├── experiment_lab/                 # Experimentation and analysis tools
├── notebooks/                      # Jupyter notebooks for analysis
├── rl_model_registry/              # Model versioning and storage
├── scripts/                        # Utility scripts (setup, data processing)
├── src/                            # Source code
│   ├── cache/                      # Runtime cache files
│   ├── graph_routing_engine/       # DRL-based routing algorithms
│   ├── fleet_management/           # Fleet optimization and EV management
│   ├── autonomous_systems/         # AV control interfaces
│   ├── demand_forecasting/         # AI demand prediction models
│   ├── traffic_prediction/         # Real-time traffic forecasting
│   ├── economic_optimization/      # Dynamic pricing and incentives
│   ├── human_interface/            # Dashboards and AR interfaces
│   └── ...                         # Additional domain modules
├── tests/                          # Test suites (unit, integration)
├── .github/workflows/              # CI/CD pipelines
├── requirements.txt                # Python dependencies
├── setup.py                        # Package configuration
└── README.md
```

## Prerequisites

To run this project, you will need:

*   **Python 3.8+**
*   **Redis**: All modules utilize Redis as a Feature Store for real-time data storage and retrieval. Ensure a Redis server is running (defaults to `localhost:6379`).
*   **Python Dependencies**:
    *   `redis`
    *   `PyYAML`
    *   `numpy`
    *   `pandas`
    *   `networkx`
    *   `asyncio` (included with Python 3.7+)
    *   `lono_libs` (for RL benchmarking and evaluation metrics)

## Quick Start

### Automated Setup
```bash
# Run the automated setup script
python scripts/setup.py --dev --test
```

### Manual Setup

1. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    **Note**: The project uses `lono_libs` for advanced RL benchmarking and evaluation metrics. This package is automatically installed from the GitHub repository as specified in `requirements.txt`. If you encounter installation issues, you can manually install it with:
    ```bash
    pip install git+https://github.com/supremeloki/lono_libs.git
    ```

2. **Start Infrastructure**
   ```bash
   # Redis (required)
   docker run -d --name redis -p 6379:6379 redis:alpine

   # Optional: Full stack with Docker Compose
   cd deployment_ops && docker-compose up -d
   ```

3. **Configure Environment**
   - Copy `conf/environments/dev.yaml` and adjust settings
   - Configure Redis connection and API endpoints

4. **Run Core Services**
   ```bash
   # Start DRL routing engine
   python src/graph_routing_engine/drl_predictive_router.py

   # Start fleet management (in another terminal)
   python src/fleet_management/ev_energy_optimizer.py
   ```

### Development Workflow

- **Testing**: `python -m pytest tests/`
- **Linting**: `flake8 src/`
- **Data Processing**: `python src/graph_routing_engine/osmnx_processor.py`
- **Experimentation**: Check `notebooks/` and `experiment_lab/`
- **RL Benchmarking**: Use `src/experimentation_tools/lono_rl_benchmark.py` for comprehensive RL agent evaluation with lono_libs integration

## Configuration

Settings for each module are located in `conf/environments/dev.yaml`. You can modify parameters such as optimization intervals, thresholds, and Redis addresses within this file.

**Example Configuration (`conf/environments/dev.yaml`):**

```yaml
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    dynamic_pricing_engine:
      enabled: true
      pricing_interval_seconds: 5
      max_price_multiplier: 3.0
      # ... other dynamic pricing engine specific settings
    av_control_interface:
      enabled: true
      telemetry_sync_interval_seconds: 2
      # ... other AV interface specific settings
    # ... settings for other modules
```

## Contribution

(Optional)
If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Test your changes.
5.  Commit your changes (`git commit -m 'Add new feature'`).
6.  Push to the branch (`git push origin feature/your-feature-name`).
7.  Create a Pull Request.

## License

(Optional)
This project is licensed under the [Your License Name] License.
