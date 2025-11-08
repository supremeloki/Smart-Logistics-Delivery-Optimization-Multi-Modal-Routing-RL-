import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailNotifier:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.email_config = self.config["environments"][environment]["alerting"].get(
            "email", {}
        )
        self.sender_email = self.email_config.get("sender_email")
        self.sender_password = self.email_config.get("sender_password")
        self.smtp_server = self.email_config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = self.email_config.get("smtp_port", 587)
        self.enabled = self.email_config.get("enabled", False)

        if not all([self.sender_email, self.sender_password, self.enabled]):
            logger.warning(
                "Email sender credentials or enablement missing. Email notifications will be disabled."
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
      email:
        enabled: false
        sender_email: your_email@example.com
        sender_password: your_app_password
        smtp_server: smtp.gmail.com
        smtp_port: 587
        recipients:
          - dev_lead@example.com
          - ops_team@example.com
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def send_email(
        self,
        subject: str,
        body: str,
        recipients: list = None,
        html_body: str = None,
        level: str = "info",
    ):
        if not self.enabled:
            logger.info(f"Email notifications are disabled. Subject: {subject}")
            return False

        target_recipients = (
            recipients if recipients else self.email_config.get("recipients", [])
        )
        if not target_recipients:
            logger.warning("No recipients specified for email notification.")
            return False

        msg = MIMEMultipart("alternative")
        msg["From"] = self.sender_email
        msg["To"] = ", ".join(target_recipients)
        msg["Subject"] = f"[{level.upper()}] {subject}"

        part1 = MIMEText(body, "plain")
        msg.attach(part1)

        if html_body:
            part2 = MIMEText(html_body, "html")
            msg.attach(part2)

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, target_recipients, msg.as_string())
            logger.info(
                f"Email notification sent successfully to {', '.join(target_recipients)} (Subject: {subject})."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    alerting:
      email:
        enabled: false # Set to true and provide real credentials to test live
        sender_email: your_email@gmail.com
        sender_password: your_app_password # Use an App Password for Gmail
        smtp_server: smtp.gmail.com
        smtp_port: 587
        recipients:
          - test_recipient@example.com
"""
            )

    notifier = EmailNotifier(config_file)

    # Test sending emails (will only work if enabled and credentials are valid)
    print("Attempting to send INFO email...")
    notifier.send_email(
        subject="System Status: All Green",
        body="The logistics system is operating perfectly with no anomalies detected.",
        level="info",
    )

    print("\nAttempting to send WARNING email with HTML content...")
    html_content = (
        """
    <html>
        <body>
            <p>Hello Team,</p>
            <p>A <b>warning</b> has been issued in the logistics system:</p>
            <p><b>Issue:</b> High pending order count observed in region X.</p>
            <p>Please investigate. Severity: <span style="color:orange;">WARNING</span></p>
            <p>Time: """
        + datetime.datetime.now().isoformat()
        + """</p>
        </body>
    </html>
    """
    )
    notifier.send_email(
        subject="Warning: Pending Order Backlog",
        body="Warning: High pending order count in region X. Please investigate.",
        html_body=html_content,
        level="warning",
    )

    print("\nAttempting to send CRITICAL email...")
    notifier.send_email(
        subject="CRITICAL: RL Agent Failure!",
        body="The RL Agent inference service is unresponsive. Immediate action required.",
        level="critical",
        recipients=["ops_oncall@example.com"],
    )
