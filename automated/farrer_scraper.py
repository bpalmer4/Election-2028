# pylint: disable=logging-fstring-interpolation
"""Farrer by-election betting scraper with error handling and notifications."""

import logging
import time
import traceback
from pathlib import Path
from typing import Dict

import pandas as pd
from bs4 import BeautifulSoup, Tag
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException

from decouple import config  # type: ignore[import-untyped]
from notifications import send_error_notification, send_success_notification


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("farrer-scraper.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Direct URL to the Farrer by-election market
MARKET_URL = (
    "https://www.sportsbet.com.au/betting/politics/"
    "australian-federal-politics./2026-farrer-by-election-10155581"
)

# Output CSV file path (relative to automated/ directory)
OUTPUT_CSV = "../betting-data/sportsbet-2026-farrer-by-election.csv"

# Notification context
SCRAPER_NAME = "Farrer By-Election Scraper"


class FarrerScraper:
    """Scraper for Farrer by-election betting odds.

    Simplified version of RobustScraper that goes directly to a known
    market URL rather than using two-step URL discovery.
    """

    def __init__(self) -> None:
        self.driver: webdriver.Chrome | None = None

        self.market_url = MARKET_URL
        self.timeout = config("SCRAPER_TIMEOUT", cast=int, default=20)
        self.retry_attempts = config("SCRAPER_RETRY_ATTEMPTS", cast=int, default=3)
        self.retry_delay = config("SCRAPER_RETRY_DELAY", cast=float, default=5.0)

        # Multi-tier fallback CSS selectors for scraping Sportsbet HTML
        self.content_selectors = [
            'div[data-automation-id="content-background"]',
            'div[data-automation-id="content"]',
            "div.content-background",
            "div.background_fja218n",
        ]

        self.outcome_selectors = [
            "div.outcomeContainer_f18v2vnr",
            "div.outcomeCardItems_f4kk892",
            'div[class*="outcome"]',
            'div[class*="card"]',
        ]

        self.name_selectors = [
            "div.nameWrapper_fddsvlq",
            'div[class*="name"]',
            'div[class*="competitor"]',
            'span[class*="name"]',
        ]

        self.price_selectors = [
            'span[class*="ButtonOddsStandardText"]',
            'button[class*="ButtonOddsStandard"]',
            "div.defaultOddsMargin_f1kcyemd",
            "div.priceText_f71sibe",
            'div[class*="price"]',
            'span[class*="price"]',
            'div[class*="odds"]',
        ]

    def setup_driver(self) -> bool:
        """Initialize Chrome WebDriver with headless configuration."""
        try:
            service = ChromeService(ChromeDriverManager().install())
            options = webdriver.ChromeOptions()
            options.add_argument("--ignore-certificate-errors")
            options.add_argument("--incognito")
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            self.driver = webdriver.Chrome(service=service, options=options)  # type: ignore[call-arg]
            self.driver.implicitly_wait(self.timeout)
            logger.info("Chrome driver setup successful")
            return True
        except Exception as e:
            logger.error(f"Failed to setup driver: {e}")
            return False

    def load_page_with_retry(self) -> BeautifulSoup | None:
        """Load the market page with automatic retry on failure."""
        for attempt in range(self.retry_attempts):
            try:
                logger.info(
                    f"Attempting to load Farrer by-election page "
                    f"(attempt {attempt + 1}/{self.retry_attempts})"
                )
                if self.driver is None:
                    raise RuntimeError("Driver not initialized")

                self.driver.get(self.market_url)
                wait = WebDriverWait(self.driver, self.timeout)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                time.sleep(2)

                soup = BeautifulSoup(self.driver.page_source, "lxml")
                logger.info("Farrer by-election page loaded successfully")
                return soup

            except (WebDriverException, TimeoutException) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All attempts to load Farrer by-election page failed")
                    return None

        return None

    def find_element_with_fallbacks(
        self,
        soup: BeautifulSoup | Tag,
        selectors: list[str],
        context: str = "",
    ) -> Tag | list[Tag] | None:
        """Try multiple CSS selectors in order until one finds elements."""
        for selector in selectors:
            try:
                if selector.startswith("div["):
                    css_elements = soup.select(selector)
                    if css_elements:
                        logger.info(f"Found {context} using CSS selector: {selector}")
                        return (
                            css_elements[0] if len(css_elements) == 1 else css_elements
                        )  # type: ignore[return-value]
                else:
                    if "data-automation-id" in selector:
                        attr_val = selector.split('"')[1]
                        find_elements = soup.find_all(
                            "div", {"data-automation-id": attr_val}
                        )
                    else:
                        class_name = selector.replace("div.", "").replace(".", " ")
                        find_elements = soup.find_all("div", {"class": class_name})

                    if find_elements:
                        logger.info(
                            f"Found {context} using fallback selector: {selector}"
                        )
                        return (
                            find_elements[0]  # type: ignore[return-value]
                            if len(find_elements) == 1
                            else find_elements  # type: ignore[return-value]
                        )

            except Exception as e:
                logger.debug(f"Selector {selector} failed for {context}: {e}")
                continue

        logger.warning(f"No working selector found for {context}")
        return None

    def extract_odds_data(self, soup: BeautifulSoup) -> Dict[str, str] | None:
        """Extract betting odds data from the market page HTML."""
        try:
            content_div = self.find_element_with_fallbacks(
                soup, self.content_selectors, "content area"
            )
            if not content_div:
                logger.error("Could not find main content area")
                return None

            content_for_search = (
                content_div[0] if isinstance(content_div, list) else content_div
            )
            outcome_containers = self.find_element_with_fallbacks(
                content_for_search,  # type: ignore[arg-type]
                self.outcome_selectors,
                "outcome containers",
            )
            if not outcome_containers:
                logger.error("Could not find outcome containers")
                return None

            if not isinstance(outcome_containers, list):
                outcome_containers = [outcome_containers]

            results = {}
            for container in outcome_containers:
                try:
                    name_element = self.find_element_with_fallbacks(
                        container,
                        self.name_selectors,
                        "name",  # type: ignore[arg-type]
                    )
                    price_element = self.find_element_with_fallbacks(
                        container,
                        self.price_selectors,
                        "price",  # type: ignore[arg-type]
                    )

                    if name_element and price_element:
                        name_tag = (
                            name_element[0]
                            if isinstance(name_element, list)
                            else name_element
                        )
                        price_tag = (
                            price_element[0]
                            if isinstance(price_element, list)
                            else price_element
                        )
                        name = name_tag.get_text(strip=True)  # type: ignore[union-attr]
                        price = price_tag.get_text(strip=True)  # type: ignore[union-attr]

                        if name and price:
                            results[name] = price
                            logger.info(f"Extracted: {name} -> {price}")

                except Exception as e:
                    logger.warning(f"Failed to extract from container: {e}")
                    continue

            if not results:
                logger.error("No odds data extracted")
                return None

            logger.info(f"Successfully extracted {len(results)} odds entries")
            return results

        except Exception as e:
            logger.error(f"Error extracting odds data: {e}")
            return None

    def save_data(self, data: Dict[str, str]) -> bool:
        """Save extracted odds data to CSV file."""
        try:
            df = pd.DataFrame([data.keys(), data.values()], index=["Party", "Odds"]).T
            df.index = pd.DatetimeIndex([pd.Timestamp.now()] * len(df))
            df.index.name = "Datetime"

            file_dir = "../betting-data"
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            file_path = OUTPUT_CSV

            file_exists = Path(file_path).exists()
            write_header = not file_exists

            if file_exists:
                try:
                    existing_df = pd.read_csv(file_path, nrows=1)
                    if (
                        len(existing_df.columns) < 3
                        or "Party" not in existing_df.columns
                    ):
                        write_header = True
                except Exception:
                    write_header = True

            df.to_csv(file_path, mode="a", index=True, header=write_header)
            logger.info(f"Data saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up Chrome WebDriver resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Chrome WebDriver cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error during WebDriver cleanup: {e}")

    def run(self) -> bool:
        """Execute the complete scraping workflow."""
        try:
            logger.info("Starting Farrer by-election scraper workflow")

            if not self.setup_driver():
                raise Exception("Failed to setup Chrome driver")

            soup = self.load_page_with_retry()
            if not soup:
                raise Exception("Failed to load market page content")

            odds_data = self.extract_odds_data(soup)
            if not odds_data:
                raise Exception("Failed to extract odds data")

            if not self.save_data(odds_data):
                raise Exception("Failed to save data")

            summary = "\n".join(
                [f"{name}: {price}" for name, price in odds_data.items()]
            )
            send_success_notification(summary, SCRAPER_NAME)

            logger.info("Farrer by-election scraper completed successfully")
            return True

        except Exception as e:
            error_msg = str(e)
            tb_info = traceback.format_exc()
            logger.error(f"Scraper workflow failed: {error_msg}")
            logger.debug(f"Full error traceback: {tb_info}")
            send_error_notification(error_msg, tb_info, SCRAPER_NAME)
            return False

        finally:
            self.cleanup()


def main():
    """Main entry point."""
    scraper = FarrerScraper()
    success = scraper.run()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
