"""Robust election betting scraper with error handling and notifications."""

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

from config import SCRAPER_CONFIG
from notifications import send_error_notification, send_success_notification


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class RobustScraper:
    """Robust scraper with fallback mechanisms and error handling."""

    def __init__(self):
        self.config = SCRAPER_CONFIG
        self.driver: webdriver.Chrome | None = None

    def setup_driver(self) -> bool:
        """Set up Chrome driver with error handling."""
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
            self.driver.implicitly_wait(self.config.timeout)

            logger.info("Chrome driver setup successful")
            return True

        except Exception as e:
            logger.error(f"Failed to setup driver: {e}")
            return False

    def get_page_content(self) -> BeautifulSoup | None:
        """Get page content with retry logic."""
        url = f"{self.config.url_base}{self.config.url_path}"

        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(
                    f"Attempting to load page (attempt {attempt + 1}/{self.config.retry_attempts})"
                )

                if self.driver is None:
                    raise Exception("Driver not initialized")

                self.driver.get(url)

                # Wait for content to load
                wait = WebDriverWait(self.driver, self.config.timeout)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                # Additional wait for dynamic content
                time.sleep(2)

                soup = BeautifulSoup(self.driver.page_source, "lxml")
                logger.info("Page loaded successfully")
                return soup

            except (WebDriverException, TimeoutException) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error("All attempts to load page failed")
                    return None

        return None

    def find_element_with_fallbacks(
        self,
        soup: BeautifulSoup | Tag,
        selectors: list[str],
        context: str = "",
    ) -> Tag | list[Tag] | None:
        """Try multiple selectors until one works."""
        for selector in selectors:
            try:
                if selector.startswith("div["):
                    # CSS selector
                    css_elements = soup.select(selector)
                    if css_elements:
                        logger.info(f"Found {context} using selector: {selector}")
                        return (
                            css_elements[0] if len(css_elements) == 1 else css_elements
                        )  # type: ignore[return-value]
                else:
                    # Class-based search
                    if "data-automation-id" in selector:
                        attr_val = selector.split('"')[1]
                        find_elements = soup.find_all(
                            "div", {"data-automation-id": attr_val}
                        )
                    else:
                        class_name = selector.replace("div.", "").replace(".", " ")
                        find_elements = soup.find_all("div", {"class": class_name})

                    if find_elements:
                        logger.info(f"Found {context} using selector: {selector}")
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
        """Extract odds data with multiple fallback strategies."""
        try:
            # Find main content area
            content_div = self.find_element_with_fallbacks(
                soup, self.config.content_selectors or [], "content area"
            )

            if not content_div:
                logger.error("Could not find main content area")
                return None

            # Find outcome containers
            # Handle content_div which could be a Tag or list[Tag]
            content_for_search = (
                content_div[0] if isinstance(content_div, list) else content_div
            )
            outcome_containers = self.find_element_with_fallbacks(
                content_for_search,  # type: ignore[arg-type]
                self.config.outcome_selectors or [],
                "outcome containers",
            )

            if not outcome_containers:
                logger.error("Could not find outcome containers")
                return None

            # Handle both single element and list
            if not isinstance(outcome_containers, list):
                outcome_containers = [outcome_containers]

            # Extract name/price pairs
            results = {}
            for container in outcome_containers:
                try:
                    # Find name
                    name_element = self.find_element_with_fallbacks(
                        container,
                        self.config.name_selectors or [],
                        "name",  # type: ignore[arg-type]
                    )

                    # Find price
                    price_element = self.find_element_with_fallbacks(
                        container,
                        self.config.price_selectors or [],
                        "price",  # type: ignore[arg-type]
                    )

                    if name_element and price_element:
                        # Handle both single elements and lists
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
        """Save data to CSV file with consistent format."""
        try:
            # Convert to DataFrame with consistent column names
            df = pd.DataFrame([data.keys(), data.values()], index=["Party", "Odds"]).T
            df.index = pd.DatetimeIndex([pd.Timestamp.now()] * len(df))
            df.index.name = "Datetime"

            # Ensure directory exists
            file_dir = "../betting-data"
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            file_path = f"{file_dir}/sportsbet-2028-election-winner.csv"

            # Check if file exists and has data
            file_exists = Path(file_path).exists()
            write_header = not file_exists

            if file_exists:
                # Check if file is empty or has inconsistent format
                try:
                    existing_df = pd.read_csv(file_path, nrows=1)
                    if (
                        len(existing_df.columns) < 3
                        or "Party" not in existing_df.columns
                    ):
                        write_header = True
                except Exception:
                    write_header = True

            # Save to file
            df.to_csv(file_path, mode="a", index=True, header=write_header)

            logger.info(f"Data saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False

    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Driver cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up driver: {e}")

    def run(self) -> bool:
        """Main scraper execution."""
        try:
            logger.info("Starting scraper run")

            # Setup driver
            if not self.setup_driver():
                raise Exception("Failed to setup Chrome driver")

            # Get page content
            soup = self.get_page_content()
            if not soup:
                raise Exception("Failed to load page content")

            # Extract data
            odds_data = self.extract_odds_data(soup)
            if not odds_data:
                raise Exception("Failed to extract odds data")

            # Save data
            if not self.save_data(odds_data):
                raise Exception("Failed to save data")

            # Send success notification (optional)
            summary = "\n".join(
                [f"{name}: {price}" for name, price in odds_data.items()]
            )
            send_success_notification(summary, "Election Betting Scraper")

            logger.info("Scraper run completed successfully")
            return True

        except Exception as e:
            error_msg = str(e)
            tb_info = traceback.format_exc()

            logger.error(f"Scraper failed: {error_msg}")
            logger.debug(f"Full traceback: {tb_info}")

            # Send error notification
            send_error_notification(error_msg, tb_info, "Election Betting Scraper")

            return False

        finally:
            self.cleanup()


def main():
    """Main entry point."""
    scraper = RobustScraper()
    success = scraper.run()

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
