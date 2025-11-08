import datetime
import yaml
import os
import logging
import numpy as np

from .slack_notifier import SlackNotifier
from .email_notifier import EmailNotifier
from ..data_access.realtime_telemetry_aggregator import (
    RealtimeTelemetryAggregator,
)  # For context
from ..utils.metrics_collector import (
    MetricsCollector,
)  # For actual metrics (optional, if pushing)
from ..stream_processing.spark_kafka_consumer import (
    SparkKafkaConsumer,
)  # For consuming anomalies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.alert_rules = self.config["environments"][environment]["alerting"]["rules"]
        self.notification_config = self.config["environments"][environment]["alerting"]

        self.slack_notifier = SlackNotifier(config_path, environment)
        self.email_notifier = EmailNotifier(config_path, environment)
        # Using SparkKafkaConsumer to read anomaly topic (if implemented to push anomalies)
        # For simplicity, we'll manually feed anomalies here, but a real system would consume from Kafka.

        self.last_alert_time = {}  # To prevent alert storms for the same rule

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    alerting:
      slack:
        enabled: false
        webhook_url: your_slack_webhook_url_here
        default_channel: #dev-alerts
      email:
        enabled: false
        sender_email: your_email@gmail.com
        sender_password: your_app_password
        smtp_server: smtp.gmail.com
        smtp_port: 587
        recipients:
          - test_recipient@example.com
      rules:
        - name: HighPendingOrders
          metric: num_pending_orders
          threshold: 50
          operator: ">"
          window_minutes: 5
          severity: warning
          cooldown_minutes: 10
          channels: ["slack", "email"]
        - name: CriticalDeliveryTimeAnomaly
          metric: avg_delivery_time_minutes
          threshold: 2.5
          operator: "z_score_gt"
          window_minutes: 1
          severity: critical
          cooldown_minutes: 5
          channels: ["slack", "email"]
        - name: DriverIdleAlert
          metric: driver_idle_percentage
          threshold: 15
          operator: ">"
          window_minutes: 10
          severity: info
          cooldown_minutes: 30
          channels: ["slack"]
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _evaluate_rule(self, rule: dict, metric_value, historical_data=None):
        operator = rule["operator"]
        threshold = rule["threshold"]

        if operator == ">":
            return metric_value > threshold
        elif operator == "<":
            return metric_value < threshold
        elif operator == "==":
            return metric_value == threshold
        elif operator == "z_score_gt":
            if historical_data is None or len(historical_data) < 10:  # Need enough data
                return False
            mean = historical_data.mean()
            std_dev = historical_data.std()
            if std_dev == 0:
                return False
            z_score = (metric_value - mean) / std_dev
            return abs(z_score) > threshold
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False

    def _check_cooldown(self, rule_name: str, cooldown_minutes: int) -> bool:
        if rule_name not in self.last_alert_time:
            return True  # No previous alert

        last_alert = self.last_alert_time[rule_name]
        time_since_last_alert = (
            datetime.datetime.now() - last_alert
        ).total_seconds() / 60
        return time_since_last_alert >= cooldown_minutes

    def _send_notifications(self, alert_info: dict, channels: list):
        message_body = (
            f"Alert '{alert_info['rule_name']}' - Severity: {alert_info['severity'].upper()}\n"
            f"Metric: {alert_info['metric_name']}, Value: {alert_info['value']:.2f}\n"
            f"Description: {alert_info['description']}"
        )

        html_body = f"""
        <html>
            <body>
                <p><b>ALERT: {alert_info['rule_name']} - {alert_info['severity'].upper()}</b></p>
                <p><b>Metric:</b> {alert_info['metric_name']}</p>
                <p><b>Current Value:</b> {alert_info['value']:.2f}</p>
                <p><b>Threshold:</b> {alert_info['operator']} {alert_info['threshold']}</p>
                <p><b>Timestamp:</b> {datetime.datetime.now().isoformat()}</p>
                <p>{alert_info['description']}</p>
            </body>
        </html>
        """

        if "slack" in channels:
            self.slack_notifier.send_notification(
                message=message_body,
                level=alert_info["severity"],
                title=f"Logistics Alert: {alert_info['rule_name']}",
            )
        if "email" in channels:
            self.email_notifier.send_email(
                subject=f"Logistics Alert: {alert_info['rule_name']}",
                body=message_body,
                html_body=html_body,
                level=alert_info["severity"],
            )

    def process_metrics_for_alerts(
        self, current_metrics: dict, historical_metrics: dict = None
    ):
        """
        Processes a batch of current metrics against defined alert rules.
        historical_metrics: A dict like {'metric_name': pd.Series_of_history_values} for Z-score calc.
        """
        for rule in self.alert_rules:
            rule_name = rule["name"]
            metric_to_check = rule["metric"]

            if metric_to_check not in current_metrics:
                continue

            current_value = current_metrics[metric_to_check]

            is_alert = False
            if rule["operator"] == "z_score_gt":
                is_alert = self._evaluate_rule(
                    rule,
                    current_value,
                    historical_data=historical_metrics.get(metric_to_check),
                )
            else:
                is_alert = self._evaluate_rule(rule, current_value)

            if is_alert:
                if self._check_cooldown(rule_name, rule["cooldown_minutes"]):
                    alert_info = {
                        "rule_name": rule_name,
                        "metric_name": metric_to_check,
                        "value": current_value,
                        "operator": rule["operator"],
                        "threshold": rule["threshold"],
                        "severity": rule["severity"],
                        "description": f"{metric_to_check} ({current_value:.2f}) exceeded threshold ({rule['operator']} {rule['threshold']:.2f}).",
                    }
                    if (
                        rule["operator"] == "z_score_gt"
                        and historical_metrics
                        and metric_to_check in historical_metrics
                    ):
                        mean = historical_metrics[metric_to_check].mean()
                        std_dev = historical_metrics[metric_to_check].std()
                        alert_info["description"] = (
                            f"{metric_to_check} ({current_value:.2f}) is an anomaly (Z-score > {rule['threshold']:.2f}) based on mean={mean:.2f}, std_dev={std_dev:.2f}."
                        )

                    self._send_notifications(alert_info, rule["channels"])
                    self.last_alert_time[rule_name] = datetime.datetime.now()
                    logger.critical(
                        f"ALERT TRIGGERED: {rule_name} (Severity: {rule['severity'].upper()})"
                    )
                else:
                    logger.info(
                        f"Alert '{rule_name}' suppressed due to cooldown period."
                    )


if __name__ == "__main__":
    # Ensure config files and necessary directories exist for local testing
    config_dir = "conf/environments"
    os.makedirs(config_dir, exist_ok=True)
    dev_config_path = os.path.join(config_dir, "dev.yaml")
    if not os.path.exists(dev_config_path):
        with open(dev_config_path, "w") as f:
            f.write(
                """
environments:
  dev:
    alerting:
      slack:
        enabled: false # Set to true and provide URL to test live
        webhook_url: your_slack_webhook_url_here
        default_channel: #dev-alerts
      email:
        enabled: false # Set to true and provide credentials to test live
        sender_email: your_email@gmail.com
        sender_password: your_app_password
        smtp_server: smtp.gmail.com
        smtp_port: 587
        recipients:
          - test_recipient@example.com
      rules:
        - name: HighPendingOrders
          metric: delivered_orders_per_minute # Using metrics from RealtimeAnomalyDetector example
          threshold: 150 # Set high to trigger
          operator: ">"
          window_minutes: 5
          severity: warning
          cooldown_minutes: 1
          channels: ["slack", "email"]
        - name: CriticalDeliveryTimeAnomaly
          metric: avg_delivery_time_minutes
          threshold: 2.5
          operator: "z_score_gt"
          window_minutes: 1
          severity: critical
          cooldown_minutes: 1
          channels: ["slack", "email"]
        - name: DriverIdleAlert
          metric: driver_idle_percentage
          threshold: 15
          operator: ">"
          window_minutes: 10
          severity: info
          cooldown_minutes: 30
          channels: ["slack"]
"""
            )

    # This example reuses the RealtimeAnomalyDetector to provide both current and historical metrics.
    # In a real scenario, the AlertManager might consume processed anomalies from Kafka, or
    # query a metrics database (e.g., Prometheus) for current and historical values.

    from data_nexus.realtime_anomaly_detector import RealtimeAnomalyDetector
    import time

    anomaly_detector = RealtimeAnomalyDetector(
        history_window_size=50, threshold_std_dev=2.5
    )
    alert_manager = AlertManager(dev_config_path)

    print("Simulating metrics stream and checking for alerts...")
    np.random.seed(42)

    for i in range(10):
        # Normal fluctuation
        delivered_orders = np.random.normal(loc=100 + i * 0.5, scale=5)
        avg_delivery_time = np.random.normal(loc=30, scale=3)
        driver_idle_time = np.random.normal(loc=5, scale=1)

        # Inject anomalies to trigger rules
        if i == 2:  # Spike in delivered_orders
            delivered_orders = (
                160  # Should trigger HighPendingOrders (if threshold 150)
            )
            print(f"\n--- Injecting high delivered_orders anomaly at step {i} ---")
        if i == 5:  # Spike in avg_delivery_time (Z-score based)
            avg_delivery_time = 50  # Will be an anomaly after some history
            print(f"\n--- Injecting high avg_delivery_time anomaly at step {i} ---")

        current_metrics = {
            "delivered_orders_per_minute": delivered_orders,
            "avg_delivery_time_minutes": avg_delivery_time,
            "driver_idle_percentage": driver_idle_time,
        }

        # Update anomaly detector's history
        anomaly_detector.process_metrics_batch(current_metrics)

        # Prepare historical data for z-score check
        historical_metrics_for_alerting = {
            m: anomaly_detector.metrics_history[m]
            for m in anomaly_detector.metrics_history
            if m in current_metrics
        }

        alert_manager.process_metrics_for_alerts(
            current_metrics, historical_metrics=historical_metrics_for_alerting
        )

        time.sleep(2)  # Simulate time passing

    print(
        "\nSimulation finished. Check console for alert messages and configured notification channels."
    )
