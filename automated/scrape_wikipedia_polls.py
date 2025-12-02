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


class URLHandler:
    """Class for handling URL requests and table extraction."""

    # HTTP configuration constants
    RETRY_ATTEMPTS = 3
    BACKOFF_FACTOR = 1
    TIMEOUT_SECONDS = 15
    RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

    def __init__(self):
        self.session = self._create_session_with_retry()

    def _create_session_with_retry(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.RETRY_ATTEMPTS,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=self.RETRY_STATUS_CODES,
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

    def get_url(self, url: str) -> str:
        """Get the text found at a URL."""
        timeout = self.TIMEOUT_SECONDS
        try:
            response = self.session.get(url, timeout=timeout)
            ensure(
                response.status_code in (200, 201, 202, 204),
                f"Failed to retrieve URL: {response.status_code}",
            )  # successful retrieval
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch URL %s: %s", url, e)
            raise

    def get_table_list(self, url: str) -> list[pd.DataFrame]:
        """Return a list of tables found at a URL. Tables are returned in pandas DataFrame format."""
        html = self.get_url(url)
        df_list = pd.read_html(StringIO(html))
        ensure(
            len(df_list) > 0, "No tables found at URL"
        )  # check we have at least one table
        return df_list


# --- DATE PARSING FUNCTIONS ---


@dataclass
class StringDateItems:
    """Data class to hold string date components."""

    day: str
    month: str
    year: str


class DateParser:
    """Class for parsing date strings from Wikipedia polling data."""

    # Date parsing constants
    ENDASH = "\u2013"
    EMDASH = "\u2014"
    HYPHEN = "\u002d"
    MINUS = "\u2212"
    COMMA = "\u002c"
    SLASH = "\u002f"
    NON_BREAKING_SPACE = "\u00a0"

    # Date validation constants
    TWO_DIGIT_YEAR_THRESHOLD = 99
    CENTURY_CUTOFF = 30
    MIN_VALID_YEAR = 2025
    MAX_VALID_YEAR = 2028

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

    @staticmethod
    def numericise(value: str) -> str:
        """Convert specific string values to rough numeric equivalents."""
        fixes = {"early": "5", "mid": "15", "late": "25"}
        return fixes.get(value, value)

    @classmethod
    def parse_date_component(
        cls, date: str, month: str = "", year: str = ""
    ) -> StringDateItems | None:
        """Parse a date string into StringDateItems."""
        if not date:
            return None

        components = date.split()
        if len(components) > 3 or len(components) < 1:
            logger.debug("Date should be in the format 'day month year'.")
            return None

        components[0] = cls.numericise(components[0].strip())
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

        components[0] = cls.numericise(components[0].strip())
        return StringDateItems(
            day=components[0], month=components[1], year=components[2]
        )

    @classmethod
    def validate_date(cls, sd: StringDateItems) -> StringDateItems | None:
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
            if y <= cls.TWO_DIGIT_YEAR_THRESHOLD:
                sd.year = f"20{sd.year}" if y < cls.CENTURY_CUTOFF else f"19{sd.year}"
                y = int(sd.year)
            if y < cls.MIN_VALID_YEAR or y > cls.MAX_VALID_YEAR:
                logger.debug(
                    "Date validation failed: year %d outside expected range (%d-%d)",
                    y,
                    cls.MIN_VALID_YEAR,
                    cls.MAX_VALID_YEAR,
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
                    rmonths = {int(v): k for k, v in cls.MONTHS.items()}
                    sd.month = rmonths[m]
                else:
                    logger.debug(
                        "Date validation failed: numeric month %d outside range 1-12", m
                    )
                    return None

            month = sd.month[:3].lower()
            if month not in cls.MONTHS:
                logger.debug(
                    "Date validation failed: month '%s' not recognized (expected: %s)",
                    sd.month,
                    list(cls.MONTHS.keys()),
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

    @classmethod
    def parse_date_range(cls, date: str) -> datetime.date | None:
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
            cls.COMMA: " ",
            cls.SLASH: " ",
            cls.NON_BREAKING_SPACE: " ",
            cls.ENDASH: "-",
            cls.EMDASH: "-",
            cls.HYPHEN: "-",
            cls.MINUS: "-",
        }
        for fix, replacement in fix_dict.items():
            date = date.replace(fix, replacement)

        date = date.replace("--", "-")
        date_list = date.split("-")
        if len(date_list) > 2 or len(date_list) < 1:
            logger.debug("Malformed date: %s", date)
            return None

        # Parse the back date
        back_date = cls.parse_date_component(date_list[-1])
        if back_date is None:
            logger.debug("Malformed date: %s", date)
            return None

        back_date = cls.validate_date(back_date)
        if back_date is None:
            logger.debug("Invalid date: %s", date)
            return None

        # Convert to datetime
        try:
            last_dt = datetime.date(
                int(back_date.year),
                int(cls.MONTHS[back_date.month]),
                int(back_date.day),
            )
        except (ValueError, KeyError) as e:
            logger.debug("Failed to create datetime: %s", e)
            return None

        if len(date_list) == 1:
            return last_dt

        front_date = cls.parse_date_component(
            date_list[0], month=back_date.month, year=back_date.year
        )
        if front_date is None:
            logger.debug("Malformed date: %s", date)
            return None

        front_date = cls.validate_date(front_date)
        if front_date is None:
            logger.debug("Invalid date: %s", date)
            return None

        try:
            first_dt = datetime.date(
                int(front_date.year),
                int(cls.MONTHS[front_date.month]),
                int(front_date.day),
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

    @classmethod
    def parse_series(cls, series: pd.Series) -> pd.Series:
        """Parse a pandas Series of date strings into datetime.date objects."""
        return series.apply(
            lambda x: cls.parse_date_range(x) if isinstance(x, str) else None
        )


class WikipediaPollingScaper:
    """Scraper for Australian federal election polling data from Wikipedia."""

    # Data validation thresholds
    PRIMARY_VOTE_LOWER = 97
    PRIMARY_VOTE_UPPER = 103
    TPP_VOTE_LOWER = 99
    TPP_VOTE_UPPER = 101
    NORMALISATION_LOWER = 99
    NORMALISATION_UPPER = 101
    ATTITUDINAL_VOTE_LOWER = 99
    ATTITUDINAL_VOTE_UPPER = 101

    @staticmethod
    def flatten_col_names(columns: pd.Index) -> list[str]:
        """Flatten the hierarchical column index."""
        ensure(columns.nlevels >= 2, "Column index must have at least 2 levels")
        flatter = [
            " ".join(col).strip() if col[0] != col[1] else col[0]
            for col in columns.values
        ]
        pattern = re.compile(r"\[.+\]")
        flat = [re.sub(pattern, "", s) for s in flatter]  # remove footnotes
        return flat

    @staticmethod
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
                    "Table %d not found (only %d tables available)",
                    table_num,
                    len(df_list),
                )
                continue
            table = df_list[table_num].copy()
            if verbose:
                logger.debug("Table %d preview:\n%s", table_num, table.head())
            flat = WikipediaPollingScaper.flatten_col_names(table.columns)
            table.columns = pd.Index(flat)
            if combined is None:
                combined = table
            else:
                table_set = set(table.columns)
                combined_set = set(combined.columns)
                problematic = table_set.difference(combined_set)
                if problematic:
                    logger.warning(
                        "With table %d, %s not in combined table.",
                        table_num,
                        problematic,
                    )
                combined = pd.concat((combined, table), ignore_index=True)

        return combined

    def __init__(self, election_year: str = "next"):
        self.url = f"https://en.wikipedia.org/wiki/Opinion_polling_for_the_{election_year}_Australian_federal_election"
        self.election_year = election_year
        self.url_handler = URLHandler()

    def fetch_page(self) -> BeautifulSoup:
        """Fetch and parse the Wikipedia page."""
        try:
            html = self.url_handler.get_url(self.url)
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
            df_list = self.url_handler.get_table_list(self.url)
            logger.info("Found %d tables on Wikipedia page", len(df_list))
            table_indices = (
                self.get_polling_table_indices(df_list, must_have)
                if table_indices is None
                else table_indices
            )
            return self.get_combined_table(df_list, table_indices, verbose=True)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to get tables: %s", e)
            return None

    def check_completeness(
        self, pattern: str, lower: int, upper: int, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Check completeness of polling data in columns selected for matching pattern.

        Logs warnings about partial/incomplete data but keeps all data."""

        columns = pd.Index(
            c
            for c in df.columns
            if pattern.lower() in c.lower() and "net" not in c.lower()
        )
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
                "Found %d rows with partial data (sum outside range [%d, %d]) for pattern '%s'",
                problematic.sum(),
                lower,
                upper,
                pattern,
            )
            logger.info("Keeping partial data for %d rows", problematic.sum())

        return df.copy()

    def distribute_undecideds(
        self, und_pattern: str, col_pattern: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Distribute undecided votes across vote columns.

        Note: the Undecideds column is only deleted if the values have been redistributed.

        IMPORTANT: This function works even when some party columns are missing (NaN).
        It redistributes undecideds proportionally across only the available columns.
        This is different from normalise_poll_data which requires ALL columns to have data.
        """

        und_cols = [
            c
            for c in df.columns
            if und_pattern.lower() in c.lower() and col_pattern.lower() in c.lower()
        ]
        if len(und_cols) != 1:
            logger.warning(
                "No unique undecided columns found matching pattern '%s' and column pattern '%s'",
                und_pattern,
                col_pattern,
            )
            return df

        und_col = und_cols[0]
        und_rows = df[und_col].notna() & (
            df[und_col] != 0
        )  # Rows where undecided column is not NaN and not zero
        if not und_rows.any():
            logger.warning("No undecided votes found in column '%s'", und_col)
            return df
        columns = [
            c for c in df.columns if col_pattern.lower() in c.lower() and c != und_col
        ]
        if not columns:
            logger.warning("No columns found matching pattern '%s'", col_pattern)
            return df

        row_sums = df[columns].sum(axis=1, skipna=True)
        und_rows = und_rows & (row_sums > 0)  # Avoid division by zero

        # redistribute - where appropriate
        for col in columns:
            df.loc[und_rows, col] += (
                df.loc[und_rows, col] / row_sums[und_rows]
            ) * df.loc[und_rows, und_col]
        return df.drop(columns=und_col).copy()  # drop the undecided column

    def normalise_poll_data(self, df: pd.DataFrame, pattern: str) -> pd.DataFrame:
        """Normalise polling data by ensuring all columns matching the pattern sum to 100%.

        Normalizes rows when:
        1. All columns have data present (no NaN values)
        2. Sum is outside the 99-101 range

        IMPORTANT: This function only operates on rows where ALL party columns have data.
        Rows with any missing values are skipped. This is different from distribute_undecideds
        which works with partial data."""

        columns = pd.Index(
            c
            for c in df.columns
            if pattern.lower() in c.lower() and "net" not in c.lower()
        )
        if len(columns) == 0:
            logger.warning("No columns found matching pattern '%s'", pattern)
            return df

        row_sums = df[columns].sum(axis=1, skipna=True)

        # Only normalize rows where ALL columns have data (no NaN values)
        all_columns_have_data = df[columns].notna().all(axis=1)

        # And the sum is outside the normalization range
        sum_outside_range = (row_sums < self.NORMALISATION_LOWER) | (
            row_sums > self.NORMALISATION_UPPER
        )

        # Normalize when both conditions are met
        problematic = all_columns_have_data & sum_outside_range

        if problematic.any():
            logger.warning(
                "Found %d rows with complete data that need to be normalised for pattern '%s'",
                problematic.sum(),
                pattern,
            )
            for col in columns:
                # Normalise each column by dividing by the row sum and multiplying by 100
                df.loc[problematic, col] = (
                    df.loc[problematic, col] / row_sums.loc[problematic] * 100
                )
        return df.copy()

    def flag_problematic_polls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag polls with incomplete data OR that don't sum to 99-101 after normalization.

        Treats missing IND (independents) values as 0 before checking completeness."""

        def check_sum(pattern: str) -> pd.Series:
            cols = [c for c in df.columns if pattern in c.lower()]
            if not cols:
                return pd.Series(False, index=df.index)

            # Make a copy and fill IND column with 0 if it exists
            df_copy = df[cols].copy()
            ind_cols = [c for c in cols if "ind" in c.lower()]
            if ind_cols:
                df_copy[ind_cols] = df_copy[ind_cols].fillna(0)

            has_any_data = df_copy.notna().any(axis=1)
            all_present = df_copy.notna().all(axis=1)
            total = df_copy.sum(axis=1)
            outside_range = (total < self.NORMALISATION_LOWER) | (
                total > self.NORMALISATION_UPPER
            )
            # Problematic if: (has data but incomplete) OR (complete but outside range)
            return (has_any_data & ~all_present) | (all_present & outside_range)

        df["problematic"] = check_sum("primary") | check_sum("2pp")
        return df.copy()

    def scrape_voting_intention_polls(
        self, table_indices: list[int] | None = None
    ) -> pd.DataFrame:
        """Scrape polling data using the simple table extraction approach."""
        raw_extracted_df = self.get_tables_simple(table_indices)

        if raw_extracted_df is None or raw_extracted_df.empty:
            logger.warning("No data extracted from tables")
            return pd.DataFrame()

        # Handle inconsistent column names - use "Firm" if "Brand" doesn't exist
        if (
            "Brand" not in raw_extracted_df.columns
            and "Firm" in raw_extracted_df.columns
        ):
            raw_extracted_df = raw_extracted_df.rename(columns={"Firm": "Brand"})

        # - delete information rows - where the Brand value is the same as the Interview mode
        # or where Brand is NaN (empty header rows)
        # (only if Interview mode column exists)
        if "Interview mode" in raw_extracted_df.columns:
            processed_df = raw_extracted_df[
                (raw_extracted_df["Brand"] != raw_extracted_df["Interview mode"])
                & (raw_extracted_df["Brand"].notna())
            ].copy()
        else:
            processed_df = raw_extracted_df[
                raw_extracted_df["Brand"].notna()
            ].copy()

        # - strip square bracket footnotes from every column
        for col in processed_df.columns:
            if processed_df[col].dtype == "object":
                processed_df[col] = (
                    processed_df[col]
                    .str.replace(r"\[[a-z0-9]+\]", "", regex=True)
                    .str.strip()
                )

        # - remove percents from all "vote" columns. and make those columns numeric
        for col in processed_df.columns:
            if ("vote" in col.lower() or "size" in col.lower()) and processed_df[
                col
            ].dtype == "object":
                processed_df[col] = (
                    processed_df[col].str.replace("%", "", regex=True).str.strip()
                )
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

        # - parse dates
        processed_df["parsed_date"] = DateParser.parse_series(processed_df["Date"])

        # - check for completeness
        processed_df = processed_df.reset_index(drop=True)
        processed_df = self.check_completeness(
            pattern="Primary",
            lower=self.PRIMARY_VOTE_LOWER,
            upper=self.PRIMARY_VOTE_UPPER,
            df=processed_df,
        )
        processed_df = self.check_completeness(
            pattern="2PP",
            lower=self.TPP_VOTE_LOWER,
            upper=self.TPP_VOTE_UPPER,
            df=processed_df,
        )

        # - redistribute undecideds
        processed_df = self.distribute_undecideds(
            und_pattern="und", col_pattern="primary", df=processed_df
        )
        processed_df = self.distribute_undecideds(
            und_pattern="und", col_pattern="2pp", df=processed_df
        )

        # - normalise
        processed_df = self.normalise_poll_data(df=processed_df, pattern="primary")
        processed_df = self.normalise_poll_data(df=processed_df, pattern="2pp")

        # - flag problematic polls that still don't sum to 100
        processed_df = self.flag_problematic_polls(df=processed_df)

        return processed_df

    def scrape_attitudinal_polls(
        self, table_indices: list[int] | None = None
    ) -> pd.DataFrame:
        """Scrape attitudinal and leadership polling data using the simple table extraction approach."""
        raw_extracted_df = self.get_tables_simple(
            table_indices, must_have=["prime minister"]
        )

        if raw_extracted_df is None or raw_extracted_df.empty:
            logger.warning("No attitudinal polling data extracted from tables")
            return pd.DataFrame()

        # Handle inconsistent column names - use "Firm" if "Brand" doesn't exist
        if (
            "Brand" not in raw_extracted_df.columns
            and "Firm" in raw_extracted_df.columns
        ):
            raw_extracted_df = raw_extracted_df.rename(columns={"Firm": "Brand"})

        # - delete information rows - where the Brand value is the same as the Interview mode
        # or where Brand is NaN (empty header rows)
        # (only if Interview mode column exists)
        if "Interview mode" in raw_extracted_df.columns:
            processed_df = raw_extracted_df[
                (raw_extracted_df["Brand"] != raw_extracted_df["Interview mode"])
                & (raw_extracted_df["Brand"].notna())
            ].copy()
        else:
            processed_df = raw_extracted_df[
                raw_extracted_df["Brand"].notna()
            ].copy()

        # - strip square bracket footnotes from every column
        for col in processed_df.columns:
            if processed_df[col].dtype == "object":
                processed_df[col] = (
                    processed_df[col]
                    .str.replace(r"\[[a-z0-9]+\]", "", regex=True)
                    .str.strip()
                )

        # - remove percents from all columns and make them numeric
        for col in processed_df.columns:
            if (
                "ley" in col.lower()
                or "albanese" in col.lower()
                or "prime minister" in col.lower()
                and processed_df[col].dtype == "object"
            ):
                processed_df[col] = (
                    processed_df[col].str.replace("%", "", regex=True).str.strip()
                )
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

        # - parse dates
        processed_df["parsed_date"] = DateParser.parse_series(processed_df["Date"])

        # - check for completeness for each group (including "don't know")
        processed_df = processed_df.reset_index(drop=True)

        # Define the three groups with specific patterns
        groups = ["Preferred prime minister", "Albanese ", "Ley "]

        # Check completeness for attitudinal polling
        for group in groups:
            processed_df = self.check_completeness(
                pattern=group,
                lower=self.ATTITUDINAL_VOTE_LOWER,
                upper=self.ATTITUDINAL_VOTE_UPPER,
                df=processed_df,
            )

        # No undecided redistribution needed for attitudinal polls

        # - normalise each group (excluding net columns)
        for group in groups:
            processed_df = self.normalise_poll_data(df=processed_df, pattern=group)

        return processed_df

    def scrape_polls(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scrape all polling data and return separate DataFrames for each type."""
        # Use the simple pandas-based approach
        logger.info("\n--- About to scrape voting intention data ---")
        vi_df = self.scrape_voting_intention_polls()
        logger.info("\n--- About to scrape attitudinal data ---")
        pm_df = self.scrape_attitudinal_polls()

        logger.info(
            "\n--- Successfully scraped %d voting intention and %d attitudinal entries ---",
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

        # Save attitudinaldata
        if not pm_df.empty:
            pm_df_to_save = pm_df.copy()

            filename = f"preferred_pm_{self.election_year}_{today}.csv"
            filepath = data_dir / filename
            pm_df_to_save.to_csv(filepath, index=False)
            logger.info("Saved attitudinaldata to %s", filepath)


# --- MAIN ---
def main():
    """Main execution function."""
    try:
        # Scrape only next election data
        elections = ["next"]

        for election in elections:
            logger.info("\n=== Scraping %s election data ===", election)
            scraper = WikipediaPollingScaper(election)
            logger.info("URL: %s", scraper.url)
            vi_df, pm_df = scraper.scrape_polls()

            if not vi_df.empty or not pm_df.empty:
                scraper.save_data(vi_df, pm_df)

                # Print summary
                if not vi_df.empty:
                    logger.info("Voting Intention - Total polls: %d", len(vi_df))
                    logger.info(
                        "Date range: %s to %s",
                        vi_df["parsed_date"].min().strftime("%Y-%m-%d"),
                        vi_df["parsed_date"].max().strftime("%Y-%m-%d"),
                    )
                    logger.info("Polling firms: %d", vi_df["Brand"].nunique())

                if not pm_df.empty:
                    logger.info("Attitudinal - Total polls: %d", len(pm_df))
                    logger.info(
                        "Date range: %s to %s",
                        pm_df["parsed_date"].min().strftime("%Y-%m-%d"),
                        pm_df["parsed_date"].max().strftime("%Y-%m-%d"),
                    )
            else:
                logger.error("No polling data found for %s", election)

    except Exception as e:
        logger.error("Script failed: %s", e)
        raise


if __name__ == "__main__":
    main()
