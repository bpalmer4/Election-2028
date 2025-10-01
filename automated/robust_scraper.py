# pylint: disable=logging-fstring-interpolation
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

from decouple import config  # type: ignore[import-untyped]
from notifications import send_error_notification, send_success_notification


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class RobustScraper:
    """Robust scraper with fallback mechanisms and error handling.

    This scraper uses a two-step process:
    1. Load the base Sportsbet politics page and scan for election market links
    2. Extract the current market URL and scrape betting odds from that specific page

    This approach makes the scraper resilient to market ID changes that occur
    every few months when Sportsbet creates new betting markets.
    """

    def __init__(self):
        # Chrome WebDriver instance - initialized later in setup_driver()
        self.driver: webdriver.Chrome | None = None

        # Configuration loaded from .env file (via python-decouple) with sensible defaults
        # These can be overridden by creating a .env file in the project root

        # Base URL containing links to all political betting markets
        self.base_url = config(
            "SCRAPER_BASE_URL",
            default="https://www.sportsbet.com.au/betting/politics/australian-federal-politics/",
            cast=str,
        )

        # Search text to find the specific election market link (case-insensitive)
        self.market_search_text = config(
            "SCRAPER_MARKET_SEARCH", default="49th parliament of australia", cast=str
        )

        # Maximum seconds to wait for page elements to load
        self.timeout = config("SCRAPER_TIMEOUT", cast=int, default=20)

        # Number of retry attempts when page loading fails
        self.retry_attempts = config("SCRAPER_RETRY_ATTEMPTS", cast=int, default=3)

        # Seconds to wait between retry attempts
        self.retry_delay = config("SCRAPER_RETRY_DELAY", cast=float, default=5.0)

        # Cached URL of the current election market (discovered during scraping)
        self.market_url: str | None = None

        # Multi-tier fallback CSS selectors for scraping Sportsbet HTML
        # These are ordered from most specific/reliable to most general
        # If Sportsbet updates their CSS classes, the scraper tries each selector until one works
        # Selectors for the main content area containing all betting markets
        self.content_selectors = [
            'div[data-automation-id="content-background"]',  # Primary: automation ID (most stable)
            'div[data-automation-id="content"]',  # Alternative automation ID
            "div.content-background",  # Fallback: semantic class name
            "div.background_fja218n",  # Last resort: generated class (fragile)
        ]

        # Selectors for individual betting outcome containers
        # Each container holds one political party/coalition and their odds
        self.outcome_selectors = [
            "div.outcomeContainer_f18v2vnr",  # Primary: specific outcome container
            "div.outcomeCardItems_f4kk892",  # Alternative: card items container
            'div[class*="outcome"]',  # Fallback: any div with "outcome" in class
            'div[class*="card"]',  # Last resort: any div with "card" in class
        ]

        # Selectors for political party/coalition names within each outcome
        self.name_selectors = [
            "div.nameWrapper_fddsvlq",  # Primary: specific name wrapper
            'div[class*="name"]',  # Fallback: any div with "name" in class
            'div[class*="competitor"]',  # Alternative: competitor containers
            'span[class*="name"]',  # Last resort: span elements with "name"
        ]

        # Selectors for betting odds/prices within each outcome
        self.price_selectors = [
            'span[class*="ButtonOddsStandardText"]',  # New selector for odds text in button
            'button[class*="ButtonOddsStandard"]',  # Button containing odds
            "div.defaultOddsMargin_f1kcyemd",  # Specific div with odds
            "div.priceText_f71sibe",  # Legacy: specific price text element
            'div[class*="price"]',  # Fallback: any div with "price" in class
            'span[class*="price"]',  # Alternative: span elements with "price"
            'div[class*="odds"]',  # Last resort: any div with "odds" in class
        ]

    def setup_driver(self) -> bool:
        """Initialize Chrome WebDriver with headless configuration and error handling.

        Returns:
            bool: True if driver setup successful, False otherwise
        """
        try:
            # Install and configure ChromeDriver automatically
            service = ChromeService(ChromeDriverManager().install())
            options = webdriver.ChromeOptions()

            # Chrome options for reliable headless operation
            options.add_argument("--ignore-certificate-errors")  # Handle SSL issues
            options.add_argument("--incognito")  # Private browsing mode
            options.add_argument("--headless")  # Run without GUI (server-friendly)
            options.add_argument(
                "--no-sandbox"
            )  # Required for some server environments
            options.add_argument(
                "--disable-dev-shm-usage"
            )  # Overcome limited resource problems
            options.add_argument("--disable-gpu")  # Disable GPU acceleration

            # Create Chrome driver instance with configured options
            self.driver = webdriver.Chrome(service=service, options=options)  # type: ignore[call-arg]

            # Set global timeout for element finding
            self.driver.implicitly_wait(self.timeout)

            logger.info("Chrome driver setup successful")
            return True

        except Exception as e:
            logger.error(f"Failed to setup driver: {e}")
            return False

    def load_page_with_retry(
        self, url: str, page_description: str
    ) -> BeautifulSoup | None:
        """Load a web page with automatic retry on failure.

        Args:
            url: The URL to load
            page_description: Human-readable description for logging

        Returns:
            BeautifulSoup object of page content, or None if all attempts failed
        """
        for attempt in range(self.retry_attempts):
            try:
                logger.info(
                    f"Attempting to load {page_description} (attempt {attempt + 1}/{self.retry_attempts})"
                )

                if self.driver is None:
                    raise RuntimeError("Driver not initialized")

                self.driver.get(url)

                # Wait for basic page structure to load
                wait = WebDriverWait(self.driver, self.timeout)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                # Additional wait for JavaScript-rendered content (betting odds are dynamic)
                time.sleep(2)

                # Parse the fully-loaded page HTML
                soup = BeautifulSoup(self.driver.page_source, "lxml")
                logger.info(f"{page_description} loaded successfully")
                return soup

            except (WebDriverException, TimeoutException) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All attempts to load {page_description} failed")
                    return None

        return None

    def find_market_url(self) -> str | None:
        """Scan the base politics page to find the current election market URL.

        This method loads the main Sportsbet politics page and searches for a link
        containing the market search text (e.g., "49th parliament of australia").
        This approach makes the scraper resilient to market ID changes.

        Returns:
            str: Full URL to the election market page, or None if not found
        """
        soup = self.load_page_with_retry(str(self.base_url), "base politics page")
        if not soup:
            return None

        # Search through all links on the page for the election market
        links = soup.find_all("a", href=True)

        for link in links:
            href = link.get("href")  # type: ignore[union-attr]
            if not href or not isinstance(href, str):
                continue

            text = link.get_text(strip=True)  # type: ignore[union-attr]
            if not isinstance(text, str):
                continue
            text_lower = text.lower()

            # Check if link text contains our search term (case-insensitive)
            if href and str(self.market_search_text).lower() in text_lower:
                # Convert relative URLs to absolute URLs
                if href.startswith("/"):
                    # Relative path: prepend domain
                    full_url = f"https://www.sportsbet.com.au{href}"
                elif href.startswith("http"):
                    # Already absolute URL
                    full_url = href
                else:
                    # Relative to current directory: append to base URL
                    full_url = f"{str(self.base_url).rstrip('/')}/{href}"

                logger.info(f"Found market link: {text} -> {full_url}")
                return full_url

        logger.error(f"No market link found containing '{self.market_search_text}'")
        return None

    def get_market_content(self) -> BeautifulSoup | None:
        """Get the HTML content of the specific election market page.

        This method first finds the market URL (if not already cached) then loads
        that specific page containing the betting odds.

        Returns:
            BeautifulSoup object of the market page, or None if failed
        """
        # Find the market URL if we haven't already
        if not self.market_url:
            self.market_url = self.find_market_url()

        # Abort if we couldn't find the market URL
        if not self.market_url:
            logger.error("Could not find market URL")
            return None

        # Load the specific market page
        return self.load_page_with_retry(self.market_url, "election market page")

    def find_element_with_fallbacks(
        self,
        soup: BeautifulSoup | Tag,
        selectors: list[str],
        context: str = "",
    ) -> Tag | list[Tag] | None:
        """Try multiple CSS selectors in order until one finds elements.

        This is the core resilience mechanism - if Sportsbet changes their CSS,
        the scraper automatically falls back to more general selectors.

        Args:
            soup: BeautifulSoup object or Tag to search within
            selectors: List of CSS selectors to try, in order of preference
            context: Description for logging (e.g., "content area", "price")

        Returns:
            Single Tag, list of Tags, or None if no selector worked
        """
        # Try each selector in order until one works
        for selector in selectors:
            try:
                if selector.startswith("div["):
                    # Modern CSS selector (preferred method)
                    css_elements = soup.select(selector)
                    if css_elements:
                        logger.info(f"Found {context} using CSS selector: {selector}")
                        return (
                            css_elements[0] if len(css_elements) == 1 else css_elements
                        )  # type: ignore[return-value]
                else:
                    # Legacy class-based search (fallback method)
                    if "data-automation-id" in selector:
                        # Extract automation ID from selector string
                        attr_val = selector.split('"')[1]
                        find_elements = soup.find_all(
                            "div", {"data-automation-id": attr_val}
                        )
                    else:
                        # Convert CSS class selector to BeautifulSoup format
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
        """Extract betting odds data from the market page HTML.

        This method uses the fallback selector system to find:
        1. The main content area containing all betting markets
        2. Individual outcome containers (one per political party)
        3. Party names and their corresponding odds within each container

        Returns:
            Dict mapping party names to their decimal odds, or None if extraction failed
        """
        try:
            # Step 1: Find the main content area containing all betting markets
            content_div = self.find_element_with_fallbacks(
                soup, self.content_selectors, "content area"
            )

            if not content_div:
                logger.error("Could not find main content area")
                return None

            # Step 2: Find individual outcome containers within the content area
            # Handle the case where content_div might be a single element or list
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

            # Ensure we have a list of containers to iterate over
            if not isinstance(outcome_containers, list):
                outcome_containers = [outcome_containers]

            # Step 3: Extract party name and odds from each outcome container
            results = {}
            for container in outcome_containers:
                try:
                    # Find the political party/coalition name within this container
                    name_element = self.find_element_with_fallbacks(
                        container,
                        self.name_selectors,
                        "name",  # type: ignore[arg-type]
                    )

                    # Find the betting odds/price within this container
                    price_element = self.find_element_with_fallbacks(
                        container,
                        self.price_selectors,
                        "price",  # type: ignore[arg-type]
                    )

                    if name_element and price_element:
                        # Handle cases where selectors return single elements or lists
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

                        # Extract clean text content (remove whitespace)
                        name = name_tag.get_text(strip=True)  # type: ignore[union-attr]
                        price = price_tag.get_text(strip=True)  # type: ignore[union-attr]

                        # Store the party -> odds mapping
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
        """Save extracted odds data to CSV file with timestamped format.

        The data is saved in the format expected by the analysis notebooks:
        - Index: datetime timestamp of when data was captured
        - Columns: Party name and decimal odds

        Args:
            data: Dictionary mapping party names to their decimal odds

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Convert dictionary to DataFrame with standardized column names
            # Transpose to get parties as rows, Party/Odds as columns
            df = pd.DataFrame([data.keys(), data.values()], index=["Party", "Odds"]).T

            # Add timestamp index - same timestamp for all parties (single scrape)
            df.index = pd.DatetimeIndex([pd.Timestamp.now()] * len(df))
            df.index.name = "Datetime"

            # Ensure the betting-data directory exists (create if needed)
            file_dir = "../betting-data"
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            file_path = f"{file_dir}/sportsbet-2028-election-winner.csv"

            # Determine if we need to write CSV headers
            file_exists = Path(file_path).exists()
            write_header = not file_exists

            if file_exists:
                # Check if existing file has correct format (headers)
                try:
                    existing_df = pd.read_csv(file_path, nrows=1)
                    if (
                        len(existing_df.columns) < 3
                        or "Party" not in existing_df.columns
                    ):
                        write_header = True  # File exists but has wrong format
                except Exception:
                    write_header = True  # File exists but can't be read

            # Append new data to CSV file
            df.to_csv(file_path, mode="a", index=True, header=write_header)

            logger.info(f"Data saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False

    def cleanup(self):
        """Clean up Chrome WebDriver resources.

        This method should always be called to prevent memory leaks and
        zombie Chrome processes.
        """
        if self.driver:
            try:
                # Properly close Chrome browser and clean up WebDriver
                self.driver.quit()
                logger.info("Chrome WebDriver cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error during WebDriver cleanup: {e}")

    def run(self) -> bool:
        """Execute the complete scraping workflow.

        This method orchestrates the entire scraping process:
        1. Set up Chrome WebDriver
        2. Find and load the current election market page
        3. Extract betting odds data
        4. Save data to CSV file
        5. Send notifications about success/failure
        6. Clean up resources

        Returns:
            bool: True if entire process successful, False if any step failed
        """
        try:
            logger.info("Starting complete scraper workflow")

            # Step 1: Initialize Chrome WebDriver
            if not self.setup_driver():
                raise Exception("Failed to setup Chrome driver")

            # Step 2: Load election market page (two-step URL discovery process)
            soup = self.get_market_content()
            if not soup:
                raise Exception("Failed to load market page content")

            # Step 3: Extract betting odds from HTML
            odds_data = self.extract_odds_data(soup)
            if not odds_data:
                raise Exception("Failed to extract odds data")

            # Step 4: Save timestamped data to CSV file
            if not self.save_data(odds_data):
                raise Exception("Failed to save data")

            # Step 5: Send success notification with scraped data summary
            summary = "\n".join(
                [f"{name}: {price}" for name, price in odds_data.items()]
            )
            send_success_notification(summary, "Election Betting Scraper")

            logger.info("Scraper run completed successfully")
            return True

        except Exception as e:
            # Handle any errors that occurred during scraping
            error_msg = str(e)
            tb_info = traceback.format_exc()

            logger.error(f"Scraper workflow failed: {error_msg}")
            logger.debug(f"Full error traceback: {tb_info}")

            # Send detailed error notification
            send_error_notification(error_msg, tb_info, "Election Betting Scraper")

            return False

        finally:
            # Always clean up WebDriver resources, even if scraping failed
            self.cleanup()


def main():
    """Main entry point when script is run directly.

    Creates a scraper instance, runs it, and exits with appropriate code.
    """
    # Create and run the scraper
    scraper = RobustScraper()
    success = scraper.run()

    # Exit with standard codes: 0 for success, 1 for failure
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
