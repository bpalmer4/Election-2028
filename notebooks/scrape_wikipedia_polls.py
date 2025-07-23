#!/usr/bin/env python3
"""
Scrape Australian federal election polling data from Wikipedia.

This script scrapes polling data from Wikipedia pages for opinion polling
on Australian federal elections, focusing on voting intention and
preferred Prime Minister data.
"""

import re
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
from io import StringIO
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- UTILITY FUNCTIONS ---

def ensure(condition: bool, message: str = "Assertion failed") -> None:
    """Simple assertion function that raises an exception if condition is False."""
    if not condition:
        raise AssertionError(message)


# --- URL HANDLING FUNCTIONS ---


def create_session_with_retry() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set default headers
    session.headers.update(
        {
            "Cache-Control": "no-cache, must-revalidate, private, max-age=0",
            "Pragma": "no-cache",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " 
            + "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
    )

    return session


def get_url(url: str, session: requests.Session | None = None) -> str:
    """Get the text found at a URL."""
    if session is None:
        session = create_session_with_retry()

    timeout = 15  # seconds
    try:
        response = session.get(url, timeout=timeout)
        ensure(
            response.status_code in (200, 201, 202, 204),
            f"Failed to retrieve URL: {response.status_code}",
        )  # successful retrieval
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch URL %s: %s", url, e)
        raise


def get_table_list(
    url: str, session: requests.Session | None = None
) -> list[pd.DataFrame]:
    """Return a list of tables found at a URL. Tables are returned in pandas DataFrame format."""
    html = get_url(url, session)
    df_list = pd.read_html(StringIO(html))
    ensure(
        len(df_list) > 0, "No tables found at URL"
    )  # check we have at least one table
    return df_list


def flatten_col_names(columns: pd.Index) -> list[str]:
    """Flatten the hierarchical column index."""
    ensure(columns.nlevels >= 2, "Column index must have at least 2 levels")
    flatter = [
        " ".join(col).strip() if col[0] != col[1] else col[0] for col in columns.values
    ]
    pattern = re.compile(r"\[.+\]")
    flat = [re.sub(pattern, "", s) for s in flatter]  # remove footnotes
    return flat


def get_combined_table(
    df_list: list[pd.DataFrame],
    table_list: list[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """Get selected tables (by int in table_list) from Wikipedia page.
    Return a single merged table for the selected tables."""

    # --- sanity check
    if table_list is None or not table_list:
        if verbose:
            logger.debug("No tables selected.")
        return None

    combined: pd.DataFrame | None = None
    for table_num in table_list:
        if table_num >= len(df_list):
            logger.warning(
                "Table %d not found (only %d tables available)", table_num, len(df_list)
            )
            continue
        table = df_list[table_num].copy()
        if verbose:
            logger.debug("Table %d preview:\n%s", table_num, table.head())
        flat = flatten_col_names(table.columns)
        table.columns = pd.Index(flat)
        if combined is None:
            combined = table
        else:
            table_set = set(table.columns)
            combined_set = set(combined.columns)
            problematic = table_set.difference(combined_set)
            if problematic:
                logger.warning(
                    "With table %d, %s not in combined table.", table_num, problematic
                )
            combined = pd.concat((combined, table), ignore_index=True)

    return combined


# --- DATE PARSING FUNCTIONS ---

# Date parsing constants
ENDASH = "\u2013"
EMDASH = "\u2014"
HYPHEN = "\u002d"
MINUS = "\u2212"
COMMA = "\u002c"
SLASH = "\u002f"
NON_BREAKING_SPACE = "\u00a0"

MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


@dataclass
class StringDateItems:
    """Data class to hold string date components."""

    day: str
    month: str
    year: str


def numericise(value: str) -> str:
    """Convert specific string values to rough numeric equivalents."""
    fixes = {"early": "5", "mid": "15", "late": "25"}
    return fixes.get(value, value)


def parse_date_component(
    date: str, month: str = "", year: str = ""
) -> StringDateItems | None:
    """Parse a date string into StringDateItems."""
    if not date:
        return None

    components = date.split()
    if len(components) > 3 or len(components) < 1:
        logger.debug("Date should be in the format 'day month year'.")
        return None

    components[0] = numericise(components[0].strip())
    if len(components) == 1:
        components = [components[0], month, year]

    if len(components) == 2 and components[0].isalpha():
        components = [components[1], components[0], year]

    if len(components) == 2:
        components = [components[0], components[1], year]

    if len(components) == 3 and components[0].isalpha():
        components = [components[1], components[0], components[2]]

    if (
        len(components) == 3
        and components[0].isdigit()
        and int(components[0]) > 31
        and components[2].isdigit()
        and int(components[2]) <= 31
    ):
        components = [components[2], components[1], components[0]]

    components[0] = numericise(components[0].strip())
    return StringDateItems(day=components[0], month=components[1], year=components[2])


def validate_date(sd: StringDateItems) -> StringDateItems | None:
    """Validate the date components."""
    # Validate year with better error messages
    if not sd.year:
        logger.debug("Date validation failed: missing year component")
        return None
    if not sd.year.isdigit():
        logger.debug("Date validation failed: year '%s' is not numeric", sd.year)
        return None

    try:
        y = int(sd.year)
        if y <= 99:
            sd.year = f"20{sd.year}" if y < 30 else f"19{sd.year}"
            y = int(sd.year)
        if y < 2020 or y > 2030:  # Broader range for polling data
            logger.debug(
                "Date validation failed: year %d outside expected range (2020-2030)", y
            )
            return None
        year = f"{y:04d}"
    except ValueError as e:
        logger.debug(
            "Date validation failed: cannot convert year '%s' to integer: %s",
            sd.year,
            e,
        )
        return None

    # Validate month with better error messages
    try:
        if sd.month.isdigit():
            m = int(sd.month)
            if 1 <= m <= 12:
                rmonths = {int(v): k for k, v in MONTHS.items()}
                sd.month = rmonths[m]
            else:
                logger.debug(
                    "Date validation failed: numeric month %d outside range 1-12", m
                )
                return None

        month = sd.month[:3].lower()
        if month not in MONTHS:
            logger.debug(
                "Date validation failed: month '%s' not recognized (expected: %s)",
                sd.month,
                list(MONTHS.keys()),
            )
            return None
    except (ValueError, AttributeError) as e:
        logger.debug(
            "Date validation failed: error processing month '%s': %s", sd.month, e
        )
        return None

    # Validate day with better error messages
    days = {
        "jan": 31,
        "feb": 28,
        "mar": 31,
        "apr": 30,
        "may": 31,
        "jun": 30,
        "jul": 31,
        "aug": 31,
        "sep": 30,
        "oct": 31,
        "nov": 30,
        "dec": 31,
    }

    try:
        # Check for leap year
        if (
            month == "feb"
            and int(year) % 4 == 0
            and (int(year) % 100 != 0 or int(year) % 400 == 0)
        ):
            days["feb"] = 29

        if not sd.day.isdigit():
            logger.debug("Date validation failed: day '%s' is not numeric", sd.day)
            return None

        d = int(sd.day)
        if d < 1 or d > days[month]:
            logger.debug(
                "Date validation failed: day %d invalid for %s (max: %d)",
                d,
                month,
                days[month],
            )
            return None
        day = f"{d:02d}"
    except (ValueError, KeyError) as e:
        logger.debug(
            "Date validation failed: error processing day '%s' for month '%s': %s",
            sd.day,
            month,
            e,
        )
        return None

    return StringDateItems(day=day, month=month, year=year)


def improved_parse_date_range(date: str) -> datetime.date | None:
    """Parse a date string into a date object using improved parsing."""
    if not date:
        return None

    date = date.strip().lower()

    # Check for ISO date format and convert to D M Y format
    iso = r"(\d{4})-(\d{2})-(\d{2})"
    while groups := re.search(iso, date):
        year, month, day = groups.groups()
        replacement = f"{day} {month} {year}"
        date = re.sub(iso, replacement, date, count=1)

    # Fix common issues with date strings
    fix_dict = {
        COMMA: " ",
        SLASH: " ",
        NON_BREAKING_SPACE: " ",
        ENDASH: "-",
        EMDASH: "-",
        HYPHEN: "-",
        MINUS: "-",
    }
    for fix, replacement in fix_dict.items():
        date = date.replace(fix, replacement)

    date = date.replace("--", "-")
    date_list = date.split("-")
    if len(date_list) > 2 or len(date_list) < 1:
        logger.debug("Malformed date: %s", date)
        return None

    # Parse the back date
    back_date = parse_date_component(date_list[-1])
    if back_date is None:
        logger.debug("Malformed date: %s", date)
        return None

    back_date = validate_date(back_date)
    if back_date is None:
        logger.debug("Invalid date: %s", date)
        return None

    # Convert to datetime
    try:
        last_dt = datetime.date(
            int(back_date.year), int(MONTHS[back_date.month]), int(back_date.day)
        )
    except (ValueError, KeyError) as e:
        logger.debug("Failed to create datetime: %s", e)
        return None

    if len(date_list) == 1:
        return last_dt

    front_date = parse_date_component(
        date_list[0], month=back_date.month, year=back_date.year
    )
    if front_date is None:
        logger.debug("Malformed date: %s", date)
        return None

    front_date = validate_date(front_date)
    if front_date is None:
        logger.debug("Invalid date: %s", date)
        return None

    try:
        first_dt = datetime.date(
            int(front_date.year), int(MONTHS[front_date.month]), int(front_date.day)
        )
    except (ValueError, KeyError) as e:
        logger.debug("Failed to create datetime: %s", e)
        return None

    if first_dt > last_dt:
        logger.debug("Invalid date range: %s", date)
        return None

    # Return the midpoint
    midpoint = first_dt + (last_dt - first_dt) / 2
    return midpoint


class WikipediaPollingScaper:
    """Scraper for Australian federal election polling data from Wikipedia."""

    # Data validation thresholds
    PRIMARY_VOTE_LOWER = 97
    PRIMARY_VOTE_UPPER = 103
    TPP_VOTE_LOWER = 99
    TPP_VOTE_UPPER = 101
    NORMALISATION_LOWER = 99.9
    NORMALISATION_UPPER = 100.1

    def __init__(self, election_year: str = "next"):
        self.url = f"https://en.wikipedia.org/wiki/Opinion_polling_for_the_{election_year}_Australian_federal_election"
        self.election_year = election_year
        self.session = create_session_with_retry()

    def fetch_page(self) -> BeautifulSoup:
        """Fetch and parse the Wikipedia page."""
        try:
            html = get_url(self.url, self.session)
            soup = BeautifulSoup(html, "html.parser")
            logger.info(
                "Successfully fetched Wikipedia page for %s", self.election_year
            )
            return soup
        except Exception as e:
            logger.error("Failed to fetch Wikipedia page: %s", e)
            raise

    def get_polling_table_indices(
        self, df_list: list[pd.DataFrame], must_have: list[str] | None = None
    ) -> list[int] | None:
        """Get indices of tables that are likely to contain polling data matching the specified patterns.

        Args:
            df_list: List of DataFrames from Wikipedia page
            must_have: List of patterns that must be present in column names (default: ["primary", "2pp"])

        Assumptions:
        - There may be some initial tables that are a prelude or introduction, and they should be skipped.
        - The next few tables are likely to be national polling data, and should be captured.
        - There will be subsequent tables that are state-specific or attitudinal polling, and should be skipped
        """

        table_indices: list[int] = []
        prelude = True
        if must_have is None:
            must_have = ["primary", "2pp"]
        for i, df in enumerate(df_list):
            # --- does this look like polling data?
            skip = False
            if not isinstance(df.columns, pd.MultiIndex):
                # expectation: polls will have multiindex columns
                skip = True
            else:
                level_0 = df.columns.get_level_values(0).str.lower()
                for check in must_have:
                    if not any(check in label for label in level_0):
                        skip = True
            if skip and prelude:
                continue  # skip any prelude tables
            if skip and not prelude:
                break  # we have finished collecting national voting intention tables

            # --- keep this table
            prelude = False
            table_indices.append(i)
            logger.info("Auto-selected table %d as a polling table", i)

        if not table_indices:
            logger.warning(
                "Could not automatically identify any national voting intention polling tables"
            )
            return None

        return table_indices

    def get_tables_simple(
        self, table_indices: list[int] | None = None, must_have: list[str] | None = None
    ) -> pd.DataFrame | None:
        """Get national polling tables using the simple approach with manual table selection."""
        try:
            df_list = get_table_list(self.url, self.session)
            logger.info("Found %d tables on Wikipedia page", len(df_list))
            table_indices = (
                self.get_polling_table_indices(df_list, must_have)
                if table_indices is None
                else table_indices
            )
            return get_combined_table(df_list, table_indices, verbose=True)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to get tables: %s", e)
            return None


    def check_completeness(self, pattern: str, lower: int, upper: int, df: pd.DataFrame) -> pd.DataFrame:
        """Check completeness of polling data in columns selected for matching pattern"""

        columns = pd.Index(c for c in df.columns if pattern.lower() in c.lower() and "net" not in c.lower())
        if len(columns) == 0:
            logger.warning("No columns found matching pattern '%s'", pattern)
            return df

        add_check = df[columns].sum(axis=1, skipna=True)

        # Only check completeness for rows that have actual data (not all NaN)
        has_data = df[columns].notna().any(axis=1)

        problematic = (add_check < lower) | (add_check > upper)
        # Only flag rows as problematic if they have data AND are outside range
        problematic = problematic & has_data

        if problematic.any():
            logger.warning(
                "Found %d rows with sum outside range [%d, %d] for pattern '%s'",
                problematic.sum(),
                lower,
                upper,
                pattern,
            )
            df.loc[problematic, columns] = np.nan  # Set problematic rows to NaN
        return df.copy()

    def distribute_undecideds(self, und_pattern: str, col_pattern: str, df: pd.DataFrame) -> pd.DataFrame:
        """Distribute undecided votes across vote columns.

        Note: the Undecideds column is only deleted if the values have been redistributed.
        """

        und_cols = [c for c in df.columns if und_pattern.lower() in c.lower() and col_pattern.lower() in c.lower()]
        if len(und_cols) != 1:
            logger.warning(
                "No unique undecided columns found matching pattern '%s' and column pattern '%s'", 
                und_pattern, col_pattern
            )
            return df

        und_col = und_cols[0]
        und_rows = df[und_col].notna() & (df[und_col] != 0)  # Rows where undecided column is not NaN and not zero
        if not und_rows.any():
            logger.warning("No undecided votes found in column '%s'", und_col)
            return df
        columns = [c for c in df.columns if col_pattern.lower() in c.lower() and c != und_col]
        if not columns:
            logger.warning("No columns found matching pattern '%s'", col_pattern)
            return df

        row_sums = df[columns].sum(axis=1, skipna=True)

        # redistribute - where appropriate
        for col in columns:
            df.loc[und_rows, col] += (df.loc[und_rows, col] / row_sums[und_rows]) * df.loc[und_rows, und_col]
        return df.drop(columns=und_col).copy()  # drop the undecided column

    def normalise_poll_data(self, df: pd.DataFrame, pattern: str) -> pd.DataFrame:
        """Normalise polling data by ensuring all columns matching the pattern sum to 100%."""

        columns = pd.Index(c for c in df.columns if pattern.lower() in c.lower() and "net" not in c.lower())
        if len(columns) == 0:
            logger.warning("No columns found matching pattern '%s'", pattern)
            return df

        row_sums = df[columns].sum(axis=1, skipna=True)

        # Only normalize rows that have actual data (not all NaN)
        has_data = df[columns].notna().any(axis=1)
        problematic = (row_sums < self.NORMALISATION_LOWER) | (row_sums > self.NORMALISATION_UPPER)
        # Only normalize rows that have data AND are outside range
        problematic = problematic & has_data

        if problematic.any():
            logger.warning(
                "Found %d rows that need to be normalised for pattern '%s'",
                problematic.sum(),
                pattern,
            )
            for col in columns:
                # Normalise each column by dividing by the row sum and multiplying by 100
                df.loc[problematic, col] = df.loc[problematic, col] / row_sums.loc[problematic] * 100
        return df.copy()

    def scrape_voting_intention_polls(
        self, table_indices: list[int] | None = None
    ) -> pd.DataFrame:
        """Scrape polling data using the simple table extraction approach."""
        raw_extracted_df = self.get_tables_simple(table_indices)

        if raw_extracted_df is None or raw_extracted_df.empty:
            logger.warning("No data extracted from tables")
            return pd.DataFrame()

        # - delete information rows - where the Brand value is the same as the Interview mode
        processed_df = raw_extracted_df[raw_extracted_df["Brand"] != raw_extracted_df["Interview mode"]].copy()

        # - strip square bracket footnotes from every column
        for col in processed_df.columns:
            if processed_df[col].dtype == "object":
                processed_df[col] = processed_df[col].str.replace(r"\[[a-z0-9]+\]", "", regex=True).str.strip()


        # - remove percents from all "vote" columns. and make those columns numeric
        for col in processed_df.columns:
            if ("vote" in col.lower() or "size" in col.lower()) and processed_df[col].dtype == "object":
                processed_df[col] = processed_df[col].str.replace("%", "", regex=True).str.strip()
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")


        # - parse dates
        processed_df["parsed_date"] = processed_df["Date"].apply(
            lambda x: improved_parse_date_range(x) if isinstance(x, str) else None
        )

        # - check for completeness
        processed_df = processed_df.reset_index(drop=True)
        processed_df = self.check_completeness(
            pattern="Primary", lower=self.PRIMARY_VOTE_LOWER, upper=self.PRIMARY_VOTE_UPPER, df=processed_df
        )
        processed_df = self.check_completeness(
            pattern="2PP", lower=self.TPP_VOTE_LOWER, upper=self.TPP_VOTE_UPPER, df=processed_df
        )

        # - redistribute undecideds
        processed_df = self.distribute_undecideds(und_pattern="und", col_pattern="primary", df=processed_df)
        processed_df = self.distribute_undecideds(und_pattern="und", col_pattern="2pp", df=processed_df)

        # - normalise
        processed_df = self.normalise_poll_data(df=processed_df, pattern="primary")
        processed_df = self.normalise_poll_data(df=processed_df, pattern="2pp")

        return processed_df

    def scrape_attitudinal_polls(
        self, table_indices: list[int] | None = None
    ) -> pd.DataFrame:
        """Scrape preferred attitudinal and leadership polling data using the simple table extraction approach."""
        raw_extracted_df = self.get_tables_simple(table_indices, must_have=["prime minister"])

        if raw_extracted_df is None or raw_extracted_df.empty:
            logger.warning("No attitudinal polling data extracted from tables")
            return pd.DataFrame()

        # Handle inconsistent column names - use "Firm" if "Brand" doesn't exist
        if "Brand" not in raw_extracted_df.columns and "Firm" in raw_extracted_df.columns:
            raw_extracted_df = raw_extracted_df.rename(columns={"Firm": "Brand"})

        # - delete information rows - where the Brand value is the same as the Interview mode
        processed_df = raw_extracted_df[raw_extracted_df["Brand"] != raw_extracted_df["Interview mode"]].copy()

        # - strip square bracket footnotes from every column
        for col in processed_df.columns:
            if processed_df[col].dtype == "object":
                processed_df[col] = processed_df[col].str.replace(r"\[[a-z0-9]+\]", "", regex=True).str.strip()

        # - remove percents from all columns and make them numeric
        for col in processed_df.columns:
            if ("ley" in col.lower() or "albanese" in col.lower() or "prime minister" in col.lower()
                and processed_df[col].dtype == "object"):
                processed_df[col] = processed_df[col].str.replace("%", "", regex=True).str.strip()
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

        # - parse dates
        processed_df["parsed_date"] = processed_df["Date"].apply(
            lambda x: improved_parse_date_range(x) if isinstance(x, str) else None
        )

        # - check for completeness for each group (including "don't know")
        processed_df = processed_df.reset_index(drop=True)

        # Define the three groups with specific patterns
        groups = ["Preferred prime minister", "Albanese ", "Ley "]

        # Check completeness for attitudinal polling
        for group in groups:
            processed_df = self.check_completeness(pattern=group, lower=90, upper=110, df=processed_df)

        # No undecided redistribution needed for attitudinal polls

        # - normalise each group (excluding net columns)
        for group in groups:
            processed_df = self.normalise_poll_data(df=processed_df, pattern=group)

        return processed_df

    def scrape_polls(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scrape all polling data and return separate DataFrames for each type."""
        # Use the simple pandas-based approach
        vi_df = self.scrape_voting_intention_polls()
        pm_df = self.scrape_attitudinal_polls()

        logger.info(
            "Successfully scraped %d voting intention and %d preferred attitudinal entries",
            len(vi_df),
            len(pm_df),
        )

        return vi_df, pm_df

    def save_data(self, vi_df: pd.DataFrame, pm_df: pd.DataFrame) -> None:
        """Save polling data to CSV files with today's date in filename."""
        # Ensure poll-data directory exists
        data_dir = Path("../poll-data")
        data_dir.mkdir(exist_ok=True)

        # Get today's date for filename
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # Save voting intention data
        if not vi_df.empty:
            vi_df_to_save = vi_df.copy()

            filename = f"voting_intention_{self.election_year}_{today}.csv"
            filepath = data_dir / filename
            vi_df_to_save.to_csv(filepath, index=False)
            logger.info("Saved voting intention data to %s", filepath)

        # Save preferred attitudinal data
        if not pm_df.empty:
            pm_df_to_save = pm_df.copy()

            filename = f"preferred_pm_{self.election_year}_{today}.csv"
            filepath = data_dir / filename
            pm_df_to_save.to_csv(filepath, index=False)
            logger.info("Saved preferred attitudinal data to %s", filepath)


# --- MAIN ---
def main():
    """Main execution function."""
    try:
        # Scrape only next election data
        elections = ["next"]

        for election in elections:
            print(f"\n=== Scraping {election} election data ===")
            scraper = WikipediaPollingScaper(election)
            print(f"URL: {scraper.url}")
            vi_df, pm_df = scraper.scrape_polls()
            print("COLUMNS: ", vi_df.columns.tolist())

            if not vi_df.empty or not pm_df.empty:
                scraper.save_data(vi_df, pm_df)

                # Print summary
                if not vi_df.empty:
                    print(f"Voting Intention - Total polls: {len(vi_df)}")
                    print(
                        f"Date range: {vi_df['parsed_date'].min().strftime('%Y-%m-%d')} to "
                        f"{vi_df['parsed_date'].max().strftime('%Y-%m-%d')}"
                    )
                    print(f"Polling firms: {vi_df['Brand'].nunique()}")

                if not pm_df.empty:
                    print(f"Attitudinal - Total polls: {len(pm_df)}")
                    print(
                        f"Date range: {pm_df['parsed_date'].min().strftime('%Y-%m-%d')} to "
                        f"{pm_df['parsed_date'].max().strftime('%Y-%m-%d')}"
                    )
            else:
                print(f"No polling data found for {election}")

    except Exception as e:
        logger.error("Script failed: %s", e)
        raise


if __name__ == "__main__":
    main()
