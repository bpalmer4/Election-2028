"""Email notification system for scraper failures."""

import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from config import EMAIL_CONFIG


def send_error_notification(
    error_message: str, 
    traceback_info: Optional[str] = None,
    scraper_name: str = "Election Scraper"
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
    if not EMAIL_CONFIG.is_configured():
        logging.warning("Email not configured - cannot send notification")
        return False

    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = EMAIL_CONFIG.sender_email
        msg["To"] = EMAIL_CONFIG.recipient_email
        msg["Subject"] = (
            f"🚨 {scraper_name} Failed - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        # Email body
        body = f"""
{scraper_name} failed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Error: {error_message}

{f"Full traceback:{traceback_info}" if traceback_info else ""}

Please check the scraper and website for changes.

---
Automated notification from Australian Federal Election 2028 scraper
        """.strip()

        msg.attach(MIMEText(body, "plain"))

        # Send email
        with smtplib.SMTP(EMAIL_CONFIG.smtp_server, EMAIL_CONFIG.smtp_port) as server:
            server.starttls()
            server.login(EMAIL_CONFIG.sender_email, EMAIL_CONFIG.sender_password)
            server.send_message(msg)

        logging.info("Error notification email sent successfully")
        return True

    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")
        return False


def send_success_notification(
    data_summary: str, 
    scraper_name: str = "Election Scraper"
) -> bool:
    """
    Send email notification when scraper succeeds (optional).

    Args:
        data_summary: Summary of captured data
        scraper_name: Name of the scraper for subject/body text

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    if not EMAIL_CONFIG.is_configured():
        return False

    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = EMAIL_CONFIG.sender_email
        msg["To"] = EMAIL_CONFIG.recipient_email
        msg["Subject"] = (
            f"✅ {scraper_name} Success - {datetime.now().strftime('%Y-%m-%d')}"
        )

        # Email body
        body = f"""
{scraper_name} completed successfully at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Data captured:
{data_summary}

---
Automated notification from Australian Federal Election 2028 scraper
        """.strip()

        msg.attach(MIMEText(body, "plain"))

        # Send email
        with smtplib.SMTP(EMAIL_CONFIG.smtp_server, EMAIL_CONFIG.smtp_port) as server:
            server.starttls()
            server.login(EMAIL_CONFIG.sender_email, EMAIL_CONFIG.sender_password)
            server.send_message(msg)

        logging.info("Success notification email sent")
        return True

    except Exception as e:
        logging.error(f"Failed to send success email: {e}")
        return False
