#!/usr/bin/env python3
"""Configuration settings for the election scraper."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


def load_env_file():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Load .env file at module import
load_env_file()


@dataclass
class EmailConfig:
    """Email configuration for notifications."""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""  # Use app password for Gmail
    recipient_email: str = ""
    
    @classmethod
    def from_env(cls) -> "EmailConfig":
        """Load email config from environment variables."""
        return cls(
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            sender_email=os.getenv("SENDER_EMAIL", ""),
            sender_password=os.getenv("SENDER_PASSWORD", ""),
            recipient_email=os.getenv("RECIPIENT_EMAIL", "")
        )
    
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.sender_email and self.sender_password and self.recipient_email)


@dataclass
class ScraperConfig:
    """Configuration for the scraper."""
    url_base: str = "https://www.sportsbet.com.au/betting/politics/australian-federal-politics/"
    url_path: str = "49th-parliament-of-australia-9232392"
    timeout: int = 10
    retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # CSS selectors with fallbacks
    content_selectors: list[str] = None
    outcome_selectors: list[str] = None
    name_selectors: list[str] = None
    price_selectors: list[str] = None
    
    def __post_init__(self):
        if self.content_selectors is None:
            self.content_selectors = [
                'div[data-automation-id="content-background"]',
                'div[data-automation-id="content"]',
                'div.content-background',
                'div.background_fja218n'
            ]
        
        if self.outcome_selectors is None:
            self.outcome_selectors = [
                'div.outcomeContainer_f18v2vnr',
                'div.outcomeCardItems_f4kk892',
                'div[class*="outcome"]',
                'div[class*="card"]'
            ]
        
        if self.name_selectors is None:
            self.name_selectors = [
                'div.nameWrapper_fddsvlq',
                'div[class*="name"]',
                'div[class*="competitor"]',
                'span[class*="name"]'
            ]
        
        if self.price_selectors is None:
            self.price_selectors = [
                'div.priceText_f71sibe',
                'div[class*="price"]',
                'span[class*="price"]',
                'div[class*="odds"]'
            ]


# Global config instances
EMAIL_CONFIG = EmailConfig.from_env()
SCRAPER_CONFIG = ScraperConfig()