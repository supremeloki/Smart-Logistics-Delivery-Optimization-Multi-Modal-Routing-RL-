import requests
import json
import datetime
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlackNotifier:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.slack_config = self.config["environments"][environment]["alerting"].get(
            "slack", {}
        )
        self.webhook_url = self.slack_config.get("webhook_url")
        self.default_channel = self.slack_config.get(
            "default_channel", "#general-alerts"
        )
        self.enabled = self.slack_config.get("enabled", False)

        if not self.webhook_url and self.enabled:
            logger.warning(
                "Slack webhook URL not configured, Slack notifications will be disabled."
            )
            self.enabled = False

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
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def send_notification(
        self, message: str, level: str = "info", channel: str = None, title: str = None
    ):
        if not self.enabled:
            logger.info(f"Slack notifications are disabled. Message: {message}")
            return

        target_channel = channel if channel else self.default_channel

        # Customize message appearance based on level
        color_map = {
            "info": "#4287f5",
            "warning": "#f5a623",
            "error": "#d0021b",
            "critical": "#8b0000",
        }
        color = color_map.get(level.lower(), "#cccccc")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title if title else f"Logistics Alert: {level.upper()}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{level.upper()} Alert:* {message}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"<!date^{int(datetime.datetime.now().timestamp())}^{{date_num}} {{time_secs}}|{datetime.datetime.now().isoformat()}>",
                    }
                ],
            },
        ]

        payload = {
            "channel": target_channel,
            "username": "Logistics-Bot",
            "icon_emoji": ":robot_face:",
            "attachments": [{"color": color, "blocks": blocks}],
        }

        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            response.raise_for_status()
            logger.info(
                f"Slack notification sent successfully to {target_channel} (Level: {level})."
            )
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


if __name__ == "__main__":
    # For a real test, replace 'https://hooks.slack.com/services/...' with an actual Slack webhook URL
    # and set 'enabled: true' in conf/environments/dev.yaml

    # Ensure dummy config for dev environment
    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    alerting:
      slack:
        enabled: false # Set to true and provide URL to test live
        webhook_url: your_slack_webhook_url_here # Replace with your actual webhook
        default_channel: #dev-alerts
"""
            )

    notifier = SlackNotifier(config_file)

    # Test sending various types of alerts
    print("Attempting to send INFO notification...")
    notifier.send_notification(
        "System is operating nominally.", level="info", title="System Status Update"
    )

    print("\nAttempting to send WARNING notification...")
    notifier.send_notification("One driver reported low fuel level.", level="warning")

    print("\nAttempting to send ERROR notification...")
    notifier.send_notification(
        "Routing API experienced a timeout for critical order ORD1234.", level="error"
    )

    print("\nAttempting to send CRITICAL notification...")
    notifier.send_notification(
        "Anomaly detected: Delivered orders dropped significantly over the last hour!",
        level="critical",
        channel="#ops-critical",
    )
