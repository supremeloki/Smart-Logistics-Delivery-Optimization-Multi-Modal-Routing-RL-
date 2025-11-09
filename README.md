# 🚚 Smart Logistics & Delivery Optimization (Multi-Modal Routing + RL)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/supremeloki/Smart-Logistics-Delivery-Optimization-Multi-Modal-Routing-RL-/actions/workflows/ci.yml/badge.svg)](https://github.com/supremeloki/Smart-Logistics-Delivery-Optimization-Multi-Modal-Routing-RL-/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-flake8-green.svg)](https://flake8.pycqa.org/)

This project implements an intelligent, multi-modal logistics optimization system using Deep Reinforcement Learning (DRL) and graph-based routing algorithms. The system optimizes delivery routes, fleet management, and real-time decision-making across urban environments, integrating AI-driven demand forecasting, dynamic pricing, and autonomous vehicle coordination.

## ✨ Key Features

- 🚗 **Multi-Modal Routing**: Combines road, drone, and autonomous vehicle routing with RL optimization
- 🔄 **Real-Time Adaptation**: Dynamic route re-planning based on traffic, weather, and driver conditions
- 🚛 **Fleet Management**: EV energy optimization, driver wellbeing monitoring, and adaptive workforce allocation
- 🏗️ **Scalable Architecture**: Modular design supporting distributed deployment and edge computing
- 📊 **Data-Driven Insights**: Integrated feature store, simulation environments, and performance monitoring
- 🧠 **Advanced RL Benchmarking**: Professional evaluation framework using lono_libs for comprehensive agent comparison
- 🐳 **Production Ready**: Docker containerization and Kubernetes deployment support
- 📈 **Monitoring & Alerting**: Integrated Slack notifications and performance metrics

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

## 📁 Project Structure

```
📂 project-root/
├── ⚙️ conf/                           # Configuration files
│   ├── 🌍 environments/               # Environment-specific configs (dev, prod)
│   ├── 🗺️ osm_processing_config.yaml   # OSM data processing settings
│   └── 🚗 routing_engine_config.yaml   # Routing engine parameters
├── 🗄️ data_nexus/                     # Data ingestion and simulation
│   ├── 🌐 raw_osm_data/                # Raw OpenStreetMap data
│   ├── 🛣️ road_network_graph/          # Processed graph data
│   └── 🎭 simulation_scenarios/        # Fleet simulation environments
├── 🚀 deployment_ops/                 # Deployment configurations
│   ├── 🐳 docker/                      # Docker containers for services
│   └── ☸️ kubernetes/                  # K8s manifests
├── 📚 docs/                           # Documentation
├── 🧪 experiment_lab/                 # Experimentation and analysis tools
├── 📓 notebooks/                      # Jupyter notebooks for analysis
├── 🤖 rl_model_registry/              # Model versioning and storage
├── 🔧 scripts/                        # Utility scripts (setup, data processing)
├── 💻 src/                            # Source code
│   ├── 💾 cache/                       # Runtime cache files
│   ├── 🧠 core/                        # Core orchestration and quantum optimization
│   ├── 🎯 core_orchestrator/           # Multi-agent RL orchestration
│   ├── 🗃️ data_access/                 # Database and telemetry management
│   ├── 🌐 data_nexus/                  # Data processing and anomaly detection
│   ├── 🚀 deployment_core/             # Deployment and inference optimization
│   ├── 👨‍🚗 driver_management/          # Driver wellbeing monitoring
│   ├── 🚁 drone_operations/            # Autonomous drone dispatching
│   ├── 💰 economic_incentive_engine/   # Dynamic pricing and incentives
│   ├── 🌐 edge_computing/              # Real-time IoT processing
│   ├── ⚖️ ethical_ai/                   # Bias detection and mitigation
│   ├── ⚙️ feature_engineering/         # Feature generation and store client
│   ├── 🏭 fleet_simulator/             # Fleet simulation utilities
│   ├── 🛣️ graph_routing/               # DRL-based graph routing
│   ├── 🏥 healthcare/                   # AI-driven diagnosis
│   ├── 🖥️ interfaces/                   # AR dashboards and AV control
│   ├── 🎓 learning/                    # Federated learning and RL training
│   ├── 🤖 ml_ops/                      # ML operations and retraining
│   ├── 📊 monitoring/                  # Alert and notification systems
│   ├── ⚙️ operations/                  # Quality control and traceability
│   ├── 💰 optimization/                # Pricing and prediction engines
│   ├── 🛣️ routing/                     # Routing algorithms and API clients
│   ├── 🏙️ routing_engine/              # Quantum path optimization
│   ├── 🏙️ smart_city/                  # Public safety systems
│   ├── 🌊 stream_processing/           # Kafka consumers and producers
│   ├── 🛠️ utils/                       # Common helpers and validators
│   └── ...                             # Additional domain modules
├── ✅ tests/                          # Test suites (unit, integration)
├── 🔄 .github/workflows/              # CI/CD pipelines
├── 📋 requirements.txt                # Python dependencies
├── 📋 requirements-dev.txt            # Development dependencies
├── 🔨 setup.py                        # Package configuration
├── 📜 LICENSE                         # MIT License
└── 📖 README.md                       # Project documentation
```

## 🛠️ Quick Start

### Prerequisites

*   🐍 **Python 3.8+**
*   🗄️ **Redis**: All modules utilize Redis as a Feature Store for real-time data storage and retrieval. Ensure a Redis server is running (defaults to `localhost:6379`).
*   📦 **Python Dependencies**:
    *   `redis`
    *   `PyYAML`
    *   `numpy`
    *   `pandas`
    *   `networkx`
    *   `asyncio` (included with Python 3.7+)
    *   `lono_libs` (for RL benchmarking and evaluation metrics - install via `pip install -r requirements-dev.txt`)

### Installation

#### Automated Setup
```bash
# Clone the repository
git clone https://github.com/supremeloki/Smart-Logistics-Delivery-Optimization-Multi-Modal-Routing-RL-.git
cd Smart-Logistics-Delivery-Optimization-Multi-Modal-Routing-RL-

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### Manual Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Running the Application

1. **Start Infrastructure**
   ```bash
   # Redis (required)
   docker run -d --name redis -p 6379:6379 redis:alpine

   # Optional: Full stack with Docker Compose
   cd deployment_ops && docker-compose up -d
   ```

2. **Configure Environment**
   - Copy `conf/environments/dev.yaml` and adjust settings
   - Configure Redis connection and API endpoints

3. **Run Core Services**
   ```bash
   # Start the main API server
   uvicorn src.deployment_core.optimization_api:app --host 0.0.0.0 --port 8001

   # Or run specific modules (in separate terminals)
   python src/routing/drl_predictive_router.py
   ```

### Development Workflow

- ✅ **Testing**: `python -m pytest tests/ -v`
- 🔍 **Linting**: `flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics`
- 📊 **Data Processing**: `python src/routing/osmnx_processor.py`
- 🧪 **Experimentation**: Check `notebooks/` and `experiment_lab/`
- 🧠 **RL Benchmarking**: Use `src/experimentation_tools/lono_rl_benchmark.py` for comprehensive RL agent evaluation with lono_libs integration
- 🐳 **Docker Build**: `docker build -f deployment_ops/docker/Dockerfile.rl_agent_service -t logistics-optimizer .`

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
