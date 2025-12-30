"""Email notification system for scraper failures."""

import smtplib
import logging
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from decouple import config  # type: ignore[import-untyped]


@dataclass
class EmailConfig:
    """Email configuration loaded from environment."""

    sender_email: str
    sender_password: str
    recipient_email: str
    smtp_server: str
    smtp_port: int

    @classmethod
    def from_env(cls) -> Optional["EmailConfig"]:
        """Load email config from environment. Returns None if not configured."""
        sender = config("SENDER_EMAIL", default="", cast=str)
        password = config("SENDER_PASSWORD", default="", cast=str)
        recipient = config("RECIPIENT_EMAIL", default="", cast=str)

        if not (sender and password and recipient):
            return None

        return cls(
            sender_email=sender,
            sender_password=password,
            recipient_email=recipient,
            smtp_server=config("SMTP_SERVER", default="smtp.gmail.com", cast=str),
            smtp_port=config("SMTP_PORT", cast=int, default=587),
        )


def _send_email(cfg: EmailConfig, subject: str, body: str) -> bool:
    """Send an email using the provided configuration."""
    try:
        msg = MIMEMultipart()
        msg["From"] = cfg.sender_email
        msg["To"] = cfg.recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(cfg.smtp_server, cfg.smtp_port) as server:
            server.starttls()
            server.login(cfg.sender_email, cfg.sender_password)
            server.send_message(msg)

        return True
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False


def send_error_notification(
    error_message: str,
    traceback_info: Optional[str] = None,
    scraper_name: str = "Election Scraper",
) -> bool:
    """
    Send email notification when scraper fails.

    Args:
        error_message: The main error message
        traceback_info: Optional full traceback information
        scraper_name: Name of the scraper for subject/body text

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    cfg = EmailConfig.from_env()
    if cfg is None:
        logging.warning("Email not configured - cannot send notification")
        return False

    subject = f"🚨 {scraper_name} Failed - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    body = f"""
{scraper_name} failed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Error: {error_message}

{f"Full traceback:{traceback_info}" if traceback_info else ""}

Please check the scraper and website for changes.

---
Automated notification from Australian Federal Election 2028 scraper
    """.strip()

    if _send_email(cfg, subject, body):
        logging.info("Error notification email sent successfully")
        return True
    return False


def send_success_notification(
    data_summary: str, scraper_name: str = "Election Scraper"
) -> bool:
    """
    Send email notification when scraper succeeds (optional).

    Args:
        data_summary: Summary of captured data
        scraper_name: Name of the scraper for subject/body text

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    cfg = EmailConfig.from_env()
    if cfg is None:
        return False

    subject = f"✅ {scraper_name} Success - {datetime.now().strftime('%Y-%m-%d')}"
    body = f"""
{scraper_name} completed successfully at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Data captured:
{data_summary}

---
Automated notification from Australian Federal Election 2028 scraper
    """.strip()

    if _send_email(cfg, subject, body):
        logging.info("Success notification email sent")
        return True
    return False
