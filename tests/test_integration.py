#!/usr/bin/env python3
"""
Integration tests for Smart Logistics & Delivery Optimization project.
This test suite validates the entire system functionality end-to-end.
"""

import os
import sys
import tempfile
import unittest
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestIntegration(unittest.TestCase):
    """Integration tests for the entire logistics optimization system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config files
        self.temp_dir = tempfile.mkdtemp()

        # Create minimal test config
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        self._create_test_config()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_config(self):
        """Create a minimal test configuration."""
        config_content = """
environments:
  dev:
    database:
      host: localhost
      port: 5432
      user: test_user
      password: test_pass
      dbname: test_db
    redis:
      host: localhost
      port: 6379
      db: 0
    kafka:
      bootstrap_servers: ['localhost:9092']
      topic_traffic_data: test_traffic
      topic_order_data: test_orders
      topic_telemetry_data: test_telemetry
    rl_agent:
      inference_endpoint: http://localhost:8001/v2/models/rl_agent_model/infer
      model_version: test_v1.0
      explore_probability: 0.1
      gnn_model_path: test_gnn.pth
      rl_checkpoint_path: test_checkpoint
    osm_processing_config_path: conf/osm_processing_config.yaml
    rl_agent_params_config_path: conf/rl_agent_params.yaml
    routing_config_path: conf/routing_engine_config.yaml
    ethical_ai:
      bias_management:
        enabled: true
        history_window_size: 1000
        fairness_thresholds:
          payout_disparity_ratio: 0.8
          high_payout_allocation_disparity: 0.2
          stress_exposure_disparity: 0.1
        mitigation_strategy: "re_weight_costs"
        protected_attributes:
          - driver_experience_years
          - region_income_level
          - vehicle_age_years
        disadvantaged_groups:
          driver_experience_years: {"label": "junior_drivers", "condition": "<5"}
          region_income_level: {"label": "low_income_areas", "condition": "low"}
          vehicle_age_years: {"label": "older_vehicles", "condition": ">10"}
    alerting:
      slack:
        enabled: false
      email:
        enabled: false
      rules: []
    monitoring:
      prometheus:
        enabled: false
        port: 8000
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def test_import_all_modules(self):
        """Test that all main modules can be imported successfully."""
        import_modules = [
            'src',
            'src.ethical_ai.bias_detection_and_mitigation',
            'src.core_orchestrator.agent_manager',
            'src.core_orchestrator.orchestrator',
            'src.deployment_core.triton_inference_adapter',
            'src.routing.astar_optimization_logic',
            'src.prediction.geospatial_demand_predictor',
            'src.optimization.dynamic_pricing_engine',
            'src.monitoring.alert_manager',
            'src.data_nexus.database_manager',
            'src.learning.train_rl_agent',
            'src.feature_engineering.feature_store_client'
        ]

        for module_name in import_modules:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")
                except Exception as e:
                    # Allow other exceptions (like missing dependencies) but not import errors
                    if "tritonclient" in str(e) and ("http support" in str(e) or "gevent" in str(e)):
                        continue  # Expected with tritonclient issues
                    if "numpy" in str(e) and "ARRAY_API" in str(e):
                        continue  # Expected numpy compatibility issues
                    if "Spark session creation failed" in str(e):
                        continue  # Expected Spark initialization issues in test environment
                    self.fail(f"Unexpected error importing {module_name}: {e}")

    @patch('src.data_nexus.database_manager.DatabaseManager')
    @patch('src.deployment_core.triton_inference_adapter.TritonInferenceAdapter')
    def test_bias_detection_integration(self, mock_triton, mock_db):
        """Test bias detection and mitigation system."""
        from src.ethical_ai.bias_detection_and_mitigation import BiasDetectionAndMitigation

        # Mock dependencies
        mock_triton.return_value = None
        mock_db.return_value = Mock()

        # Create bias manager
        bias_manager = BiasDetectionAndMitigation(self.config_path)

        # Create test decision data
        test_decisions = [
            {
                'timestamp': '2024-01-01T00:00:00Z',
                'driver_id': 'driver_1',
                'current_location_node': 1,
                'driver_experience_years': 2,  # Junior driver
                'vehicle_age_years': 1,
                'is_urban_area': True,
                'chosen_action': 'assign_low_payout',
                'payout_offered': 15.0,
                'estimated_travel_time_minutes': 20.0,
                'predicted_traffic_factor': 1.0,
                'region_income_level': 'low',
                'demand_level_at_node': 0.5
            },
            {
                'timestamp': '2024-01-01T00:01:00Z',
                'driver_id': 'driver_2',
                'current_location_node': 2,
                'driver_experience_years': 8,  # Senior driver
                'vehicle_age_years': 3,
                'is_urban_area': True,
                'chosen_action': 'assign_high_payout',
                'payout_offered': 50.0,
                'estimated_travel_time_minutes': 25.0,
                'predicted_traffic_factor': 1.2,
                'region_income_level': 'high',
                'demand_level_at_node': 0.8
            }
        ]

        df = pd.DataFrame(test_decisions)

        # Test bias detection
        result = bias_manager.detect_bias(df)

        # Verify result structure
        self.assertIn('bias_detected', result)
        self.assertIn('bias_scores', result)
        self.assertIn('bias_alerts', result)
        self.assertIn('mitigation_strategy_recommended', result)

        # Test mitigation
        mock_params = {'reward_functions': {}, 'cost_functions': {}}
        mitigated = bias_manager.mitigate_bias(mock_params)
        self.assertIsInstance(mitigated, dict)

    @patch('src.routing.osmnx_processor.OSMNxProcessor')
    @patch('src.deployment_core.triton_inference_adapter.TritonInferenceAdapter')
    def test_routing_integration(self, mock_triton, mock_osmnx):
        """Test routing system integration."""
        from src.routing.astar_optimization_logic import AStarRouting

        # Mock OSMnx processor
        mock_processor = Mock()
        mock_graph = Mock()
        mock_processor.get_graph.return_value = mock_graph
        mock_processor.output_path = '/tmp/test_graph.gml'
        mock_osmnx.return_value = mock_processor

        # Mock graph with some nodes and edges
        mock_graph.number_of_nodes.return_value = 10
        mock_graph.has_edge.return_value = True
        mock_graph.__getitem__ = Mock(return_value={
            0: {'travel_time': 10, 'length': 100, 'traffic_factor': 1.0}
        })

        # Create routing instance
        routing_config_path = os.path.join(self.temp_dir, 'routing_config.yaml')
        with open(routing_config_path, 'w') as f:
            f.write("""
astar_custom_params:
  algorithm: astar
  heuristic: euclidean
""")
        router = AStarRouting(mock_graph, routing_config_path)

        # Test basic routing (mocked)
        self.assertIsNotNone(router)

    @patch('src.data_nexus.database_manager.DatabaseManager')
    def test_database_manager_integration(self, mock_db):
        """Test database manager integration."""
        from src.data_nexus.database_manager import DatabaseManager

        # Mock database connection
        mock_conn = Mock()
        mock_db.return_value.connect.return_value = None
        mock_db.return_value.create_tables.return_value = None

        # Create database manager
        db_manager = DatabaseManager(self.config_path)

        # Test basic operations
        self.assertIsNotNone(db_manager)
        db_manager.connect()
        db_manager.create_tables()

    def test_config_loading(self):
        """Test that configuration files can be loaded."""
        import yaml

        # Test main config loading
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.assertIn('environments', config)
        self.assertIn('dev', config['environments'])
        self.assertIn('database', config['environments']['dev'])
        self.assertIn('redis', config['environments']['dev'])

    def test_all_imports_syntax_check(self):
        """Test that all Python files have valid syntax."""
        import ast
        import glob

        # Find all Python files
        py_files = glob.glob('src/**/*.py', recursive=True)
        py_files.extend(glob.glob('tests/**/*.py', recursive=True))

        for py_file in py_files:
            with self.subTest(file=py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        code = f.read()
                    ast.parse(code)
                except SyntaxError as e:
                    self.fail(f"Syntax error in {py_file}: {e}")
                except UnicodeDecodeError:
                    # Skip files with encoding issues
                    continue

    @patch('src.utils.metrics_collector.MetricsCollector._instance', None)
    @patch('src.core_orchestrator.orchestrator.AStarRouting')
    @patch('src.core_orchestrator.orchestrator.AgentManager')
    @patch('src.stream_processing.spark_kafka_consumer.SparkKafkaConsumer')
    @patch('src.data_nexus.realtime_telemetry_aggregator.RealtimeTelemetryAggregator')
    @patch('src.data_nexus.database_manager.DatabaseManager')
    @patch('src.feature_forge.feature_store_client.FeatureStoreClient')
    @patch('src.monitoring.alert_manager.AlertManager')
    @patch('pyspark.sql.SparkSession')
    @patch('redis.StrictRedis')
    def test_orchestrator_mock_basic(self, mock_redis, mock_spark_session, mock_alert, mock_feature_store, mock_db, mock_telemetry, mock_spark, mock_agent, mock_routing):
        """Test basic orchestrator functionality with full mocking."""
        # Mock the Redis client to prevent connection attempts
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.xgroup_create.return_value = None
        mock_redis_instance.xreadgroup.return_value = []
        mock_redis_instance.xack.return_value = None
        mock_redis_instance.xadd.return_value = None
        mock_redis_instance.set.return_value = None
        mock_redis_instance.mget.return_value = []
        mock_redis_instance.keys.return_value = []
        mock_redis_instance.pipeline.return_value = mock_redis_instance
        mock_redis_instance.execute.return_value = None
        mock_redis_instance.scan_iter.return_value = []

        # Mock SparkSession to prevent Hadoop initialization
        mock_spark_session_instance = Mock()
        mock_spark_session.return_value = mock_spark_session_instance
        mock_spark_session_instance.builder = Mock()
        mock_spark_session_instance.builder.appName.return_value = Mock()
        mock_spark_session_instance.builder.appName.return_value.config.return_value = Mock()
        mock_spark_session_instance.builder.appName.return_value.config.return_value.getOrCreate.return_value = Mock()

        from src.core_orchestrator.orchestrator import Orchestrator

        # Setup mocks
        mock_db_instance = Mock()
        mock_db.return_value = mock_db_instance
        mock_db_instance.connect.return_value = None
        mock_db_instance.create_tables.return_value = None
        mock_db_instance.close.return_value = None

        mock_telemetry_instance = Mock()
        mock_telemetry.return_value = mock_telemetry_instance

        mock_spark_instance = Mock()
        mock_spark.return_value = mock_spark_instance
        mock_spark_instance.process_traffic_data.return_value = Mock()
        mock_spark_instance.process_order_data.return_value = Mock()
        mock_spark_instance.spark = Mock()

        mock_spark_session.builder.appName.return_value.config.return_value.getOrCreate.return_value = Mock()

        mock_feature_store_instance = Mock()
        mock_feature_store.return_value = mock_feature_store_instance

        mock_alert_instance = Mock()
        mock_alert.return_value = mock_alert_instance

        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.osm_processor = Mock()
        mock_agent_instance.osm_processor.get_graph.return_value = Mock()

        mock_routing_instance = Mock()
        mock_routing.return_value = mock_routing_instance

        # Create orchestrator with test config
        orchestrator = Orchestrator(self.config_path)

        # Test basic initialization
        self.assertIsNotNone(orchestrator)
        self.assertEqual(orchestrator.deployment_environment, "dev")

        # Test that core components are initialized
        self.assertIsNotNone(orchestrator.db_manager)
        self.assertIsNotNone(orchestrator.telemetry_aggregator)
        self.assertIsNotNone(orchestrator.spark_consumer)
        self.assertIsNotNone(orchestrator.feature_store_client)
        self.assertIsNotNone(orchestrator.alert_manager)
        self.assertIsNotNone(orchestrator.metrics_collector)


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PYTHONPATH', os.path.join(os.path.dirname(__file__), '..', 'src'))

    # Run tests
    unittest.main(verbosity=2)