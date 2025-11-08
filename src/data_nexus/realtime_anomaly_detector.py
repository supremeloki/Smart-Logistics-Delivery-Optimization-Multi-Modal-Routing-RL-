import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeAnomalyDetector:
    """
    A simple real-time anomaly detector that maintains a rolling history of metrics
    and can detect anomalies based on statistical thresholds.
    """

    def __init__(self, history_window_size=100, threshold_std_dev=2.0):
        """
        Initialize the anomaly detector.

        Args:
            history_window_size (int): Maximum number of historical values to keep per metric.
            threshold_std_dev (float): Number of standard deviations for anomaly threshold.
        """
        self.history_window_size = history_window_size
        self.threshold_std_dev = threshold_std_dev
        self.metrics_history = {}  # dict of metric_name -> list of values

    def process_metrics_batch(self, current_metrics: dict):
        """
        Process a batch of current metrics, adding them to the history.

        Args:
            current_metrics (dict): Dictionary of metric_name -> current_value
        """
        for metric_name, value in current_metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []

            # Add new value
            self.metrics_history[metric_name].append(value)

            # Maintain window size
            if len(self.metrics_history[metric_name]) > self.history_window_size:
                self.metrics_history[metric_name].pop(0)

    def detect_anomalies(self, current_metrics: dict):
        """
        Detect anomalies in current metrics based on historical data.

        Args:
            current_metrics (dict): Dictionary of metric_name -> current_value

        Returns:
            dict: Dictionary of metric_name -> (is_anomaly, z_score)
        """
        anomalies = {}

        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < 10:
                # Not enough data for statistical analysis
                anomalies[metric_name] = (False, 0.0)
                continue

            history = self.metrics_history[metric_name]
            mean = np.mean(history)
            std_dev = np.std(history)

            if std_dev == 0:
                z_score = 0.0
            else:
                z_score = (current_value - mean) / std_dev

            is_anomaly = abs(z_score) > self.threshold_std_dev
            anomalies[metric_name] = (is_anomaly, z_score)

        return anomalies

    def get_historical_series(self, metric_name):
        """
        Get historical values as a pandas Series for a metric.

        Args:
            metric_name (str): Name of the metric

        Returns:
            pd.Series: Historical values
        """
        if metric_name not in self.metrics_history:
            return pd.Series()

        return pd.Series(self.metrics_history[metric_name])


if __name__ == "__main__":
    # Example usage
    detector = RealtimeAnomalyDetector(history_window_size=50, threshold_std_dev=2.5)

    # Simulate some metrics
    np.random.seed(42)
    for i in range(30):
        metrics = {
            "cpu_usage": np.random.normal(50, 10),
            "memory_usage": np.random.normal(60, 5),
            "response_time": np.random.normal(200, 50)
        }

        detector.process_metrics_batch(metrics)

        if i > 10:  # After some history
            anomalies = detector.detect_anomalies(metrics)
            anomalous_metrics = [k for k, (is_anom, _) in anomalies.items() if is_anom]
            if anomalous_metrics:
                logger.info(f"Step {i}: Anomalies detected in {anomalous_metrics}")

    # Inject an anomaly
    anomalous_metrics = {"cpu_usage": 90, "memory_usage": 80, "response_time": 500}
    detector.process_metrics_batch(anomalous_metrics)
    anomalies = detector.detect_anomalies(anomalous_metrics)
    anomalous_metrics_names = [k for k, (is_anom, _) in anomalies.items() if is_anom]
    logger.info(f"Injected anomaly detection: {anomalous_metrics_names}")