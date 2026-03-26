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
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from utils import ensure

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    def get_table_list(
        self, url: str, header: list[int] | int = [0, 1]
    ) -> list[pd.DataFrame]:
        """Return a list of tables found at a URL. Tables are returned in pandas DataFrame format."""
        html = self.get_url(url)
        # Filter to wikitable-class tables only to avoid parsing
        # navigation/notice tables that may have fewer header rows
        soup = BeautifulSoup(html, "html.parser")
        wikitables = soup.find_all("table", class_="wikitable")
        df_list = []
        self._table_years: dict[int, str] = {}
        for table in wikitables:
            idx = len(df_list)
            df_list.extend(pd.read_html(StringIO(str(table)), header=header))
            # Extract year from preceding heading (e.g. "2026", "2025")
            prev_heading = table.find_previous(["h2", "h3", "h4"])
            if prev_heading:
                heading_text = prev_heading.get_text().strip()
                year_match = re.match(r"^(\d{4})$", heading_text)
                if year_match:
                    self._table_years[idx] = year_match.group(1)
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

    # Month info: (month_number, days_in_month)
    MONTH_INFO: dict[str, tuple[str, int]] = {
        "jan": ("01", 31),
        "feb": ("02", 28),
        "mar": ("03", 31),
        "apr": ("04", 30),
        "may": ("05", 31),
        "jun": ("06", 30),
        "jul": ("07", 31),
        "aug": ("08", 31),
        "sep": ("09", 30),
        "oct": ("10", 31),
        "nov": ("11", 30),
        "dec": ("12", 31),
    }
    MONTHS = {k: v[0] for k, v in MONTH_INFO.items()}

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
    def _is_leap_year(cls, year: int) -> bool:
        """Check if a year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @classmethod
    def validate_date(cls, sd: StringDateItems) -> StringDateItems | None:
        """Validate the date components."""
        try:
            # Validate year
            if not sd.year or not sd.year.isdigit():
                return None
            y = int(sd.year)
            if y <= cls.TWO_DIGIT_YEAR_THRESHOLD:
                y = int(f"20{sd.year}" if y < cls.CENTURY_CUTOFF else f"19{sd.year}")
            if not cls.MIN_VALID_YEAR <= y <= cls.MAX_VALID_YEAR:
                return None
            year = f"{y:04d}"

            # Validate month
            if sd.month.isdigit():
                m = int(sd.month)
                if not 1 <= m <= 12:
                    return None
                rmonths = {int(v[0]): k for k, v in cls.MONTH_INFO.items()}
                sd.month = rmonths[m]
            month = sd.month[:3].lower()
            if month not in cls.MONTH_INFO:
                return None

            # Validate day
            if not sd.day.isdigit():
                return None
            d = int(sd.day)
            max_days = cls.MONTH_INFO[month][1]
            if month == "feb" and cls._is_leap_year(y):
                max_days = 29
            if not 1 <= d <= max_days:
                return None

            return StringDateItems(day=f"{d:02d}", month=month, year=year)
        except (ValueError, KeyError, AttributeError):
            return None

    @classmethod
    def _to_date(cls, sd: StringDateItems) -> datetime.date | None:
        """Convert validated StringDateItems to datetime.date."""
        try:
            return datetime.date(int(sd.year), int(cls.MONTHS[sd.month]), int(sd.day))
        except (ValueError, KeyError):
            return None

    @classmethod
    def _parse_validate_convert(
        cls, date_str: str, month: str = "", year: str = ""
    ) -> datetime.date | None:
        """Parse, validate, and convert a date string to datetime.date."""
        parsed = cls.parse_date_component(date_str, month, year)
        if parsed is None:
            return None
        validated = cls.validate_date(parsed)
        if validated is None:
            return None
        return cls._to_date(validated)

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
            date = re.sub(iso, f"{day} {month} {year}", date, count=1)

        # Normalize separators
        for char in (cls.COMMA, cls.SLASH, cls.NON_BREAKING_SPACE):
            date = date.replace(char, " ")
        for char in (cls.ENDASH, cls.EMDASH, cls.HYPHEN, cls.MINUS):
            date = date.replace(char, "-")

        date = date.replace("--", "-")
        date_list = date.split("-")
        if not 1 <= len(date_list) <= 2:
            return None

        # Parse and validate back date (end of range)
        back = cls.parse_date_component(date_list[-1])
        if back is None or (back := cls.validate_date(back)) is None:
            return None
        last_dt = cls._to_date(back)
        if last_dt is None:
            return None

        if len(date_list) == 1:
            return last_dt

        # Parse front date using back date's month/year as defaults
        first_dt = cls._parse_validate_convert(date_list[0], back.month, back.year)
        if first_dt is None or first_dt > last_dt:
            return None

        return first_dt + (last_dt - first_dt) / 2

    @classmethod
    def parse_series(cls, series: pd.Series) -> pd.Series:
        """Parse a pandas Series of date strings into datetime.date objects."""
        return series.apply(
            lambda x: cls.parse_date_range(x) if isinstance(x, str) else None
        )


class WikipediaPollingScaper:
    """Scraper for Australian federal election polling data from Wikipedia."""

    # Data validation thresholds
    PRIMARY_VOTE_LOWER = 98
    PRIMARY_VOTE_UPPER = 102
    TPP_VOTE_LOWER = 99
    TPP_VOTE_UPPER = 101
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
    def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Merge columns that are pandas auto-deduplicated copies (e.g. 'X.1', 'X.2').

        This handles the case where Wikipedia splits a column into sub-columns
        (e.g. L/NP split into LIB, LNP, NAT). When parsed with pd.read_html,
        colspan causes identical values across sub-columns, while split values
        differ. This method sums the differing values and keeps the single value
        when they're identical, then drops the duplicate columns.
        """
        # Find columns with .N suffixes that have a base column
        suffix_pattern = re.compile(r"^(.+)\.\d+$")
        base_to_dupes: dict[str, list[str]] = {}
        for col in df.columns:
            m = suffix_pattern.match(col)
            if m:
                base = m.group(1)
                if base in df.columns:
                    base_to_dupes.setdefault(base, []).append(col)

        df = df.copy()

        if not base_to_dupes:
            return WikipediaPollingScaper._fix_colspan_ind_oth(df)

        cols_to_drop: list[str] = []
        for base, dupes in base_to_dupes.items():
            all_cols = [base] + dupes
            logger.info(
                "Merging duplicate columns: %s -> %s", all_cols, base
            )
            # Strip % signs and convert to numeric for comparison
            numeric_df = pd.DataFrame({
                c: pd.to_numeric(
                    df[c].astype(str).str.replace("%", "", regex=False).str.strip(),
                    errors="coerce",
                )
                for c in all_cols
            })

            # Colspan makes multiple sub-columns share the same value.
            # For each row, deduplicate identical values then sum the distinct ones.
            # e.g. LIB=21.5, LNP=21.5, NAT=4 -> unique values 21.5+4 = 25.5
            def _row_merge(row: pd.Series) -> float:
                non_nan = row.dropna().unique()
                return float(non_nan.sum()) if len(non_nan) > 0 else np.nan

            df[base] = numeric_df.apply(_row_merge, axis=1)
            cols_to_drop.extend(dupes)

        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped duplicate columns: %s", cols_to_drop)
        return WikipediaPollingScaper._fix_colspan_ind_oth(df)

    @staticmethod
    def _fix_colspan_ind_oth(df: pd.DataFrame) -> pd.DataFrame:
        """Fix data-row colspans between IND/OTH columns.

        When a data cell has colspan=2, pd.read_html fills both columns
        with the same value. Detect this by checking if IND == OTH
        AND the primary sum exceeds 102%.
        """
        vote_cols = [c for c in df.columns if "primary vote" in c.lower()]
        ind_col = next((c for c in vote_cols if "ind" in c.lower()), None)
        oth_col = next((c for c in vote_cols if "oth" in c.lower()), None)
        if not (ind_col and oth_col):
            return df

        def _clean_numeric(series: pd.Series) -> pd.Series:
            """Strip %, footnotes, and whitespace then convert to numeric."""
            return pd.to_numeric(
                series.astype(str)
                .str.replace(r"\[.*?\]", "", regex=True)
                .str.replace("%", "", regex=False)
                .str.strip(),
                errors="coerce",
            )

        ind_num = _clean_numeric(df[ind_col])
        oth_num = _clean_numeric(df[oth_col])
        vote_numeric = pd.DataFrame({c: _clean_numeric(df[c]) for c in vote_cols})
        row_sum = vote_numeric.sum(axis=1, skipna=True)
        colspan_mask = (ind_num == oth_num) & ind_num.notna() & (row_sum > 102)
        if colspan_mask.any():
            logger.info(
                "Clearing %d OTH values duplicated from IND by colspan",
                colspan_mask.sum(),
            )
            df.loc[colspan_mask, oth_col] = np.nan

        return df

    @staticmethod
    def fix_satisfaction_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Fix satisfaction table columns that have 'Unnamed' in level 1.

        Wikipedia satisfaction tables have sub-headers (Pos./Neg./DK/Net) in row 0,
        not in the column multi-index. This method renames columns using those values.
        """
        # Map Wikipedia sub-headers to our standard column names
        subheader_map = {
            "pos.": "Satisfied",
            "neg.": "Dissatisfied",
            "dk": "Don't know",
            "net": "Net",
        }

        # Check if we have "Unnamed" columns that need fixing
        unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
        if not unnamed_cols:
            return df

        # The first data row might contain sub-headers like "Pos.", "Neg.", etc.
        # Check the first few rows to find the sub-header row
        for row_idx in range(min(3, len(df))):
            row = df.iloc[row_idx]
            # Check if this row has sub-header values
            has_subheaders = any(
                str(v).lower().strip().rstrip(".") in ["pos", "neg", "dk", "net"]
                for v in row.values
                if pd.notna(v)
            )
            if has_subheaders:
                # Build column rename mapping
                renames = {}
                for col in unnamed_cols:
                    val = str(row[col]).lower().strip() if pd.notna(row[col]) else ""
                    if val in subheader_map:
                        # Extract leader name from column (e.g., "Albanese Unnamed: 3_level_1" -> "Albanese")
                        leader = col.split()[0] if " " in col else col
                        renames[col] = f"{leader} {subheader_map[val]}"

                if renames:
                    df = df.rename(columns=renames)
                    logger.info(
                        "Fixed %d satisfaction columns from row %d",
                        len(renames),
                        row_idx,
                    )
                    # Drop the sub-header row and any duplicate header rows
                    df = df.iloc[row_idx + 1 :].reset_index(drop=True)
                break

        return df

    @staticmethod
    def get_combined_table(
        df_list: list[pd.DataFrame],
        table_list: list[int] | None = None,
        verbose: bool = False,
        table_years: dict[int, str] | None = None,
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
            table = WikipediaPollingScaper.merge_duplicate_columns(table)

            # Append year to Date column if dates lack year info
            if table_years and table_num in table_years and "Date" in table.columns:
                year = table_years[table_num]
                # Only append year to date strings that don't already contain a 4-digit year
                year_pattern = re.compile(r"\d{4}")
                table["Date"] = table["Date"].apply(
                    lambda d, y=year: (
                        f"{d} {y}"
                        if isinstance(d, str) and not year_pattern.search(d)
                        else d
                    )
                )
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
        self,
        df_list: list[pd.DataFrame],
        must_have: list[str] | None = None,
        any_of: list[str] | None = None,
    ) -> list[int] | None:
        """Get indices of tables that are likely to contain polling data matching the specified patterns.

        Args:
            df_list: List of DataFrames from Wikipedia page
            must_have: List of patterns that must ALL be present in column names (AND logic)
            any_of: List of patterns where ANY can be present (OR logic) - used instead of must_have

        Assumptions:
        - There may be some initial tables that are a prelude or introduction, and they should be skipped.
        - The next few tables are likely to be national polling data, and should be captured.
        - There will be subsequent tables that are state-specific or attitudinal polling, and should be skipped
        """

        table_indices: list[int] = []
        prelude = True
        if must_have is None and any_of is None:
            must_have = ["primary", "2pp"]
        for i, df in enumerate(df_list):
            # --- does this look like polling data?
            skip = False
            if not isinstance(df.columns, pd.MultiIndex):
                # expectation: polls will have multiindex columns
                skip = True
            else:
                level_0 = df.columns.get_level_values(0).str.lower()
                if any_of is not None:
                    # OR logic: skip if NONE of the patterns match
                    if not any(
                        any(pattern in label for label in level_0) for pattern in any_of
                    ):
                        skip = True
                elif must_have is not None:
                    # AND logic: skip if ANY pattern doesn't match
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
        self,
        table_indices: list[int] | None = None,
        must_have: list[str] | None = None,
        any_of: list[str] | None = None,
    ) -> pd.DataFrame | None:
        """Get national polling tables using the simple approach with manual table selection."""
        try:
            df_list = self.url_handler.get_table_list(self.url)
            table_years = getattr(self.url_handler, "_table_years", {})
            logger.info("Found %d tables on Wikipedia page", len(df_list))
            table_indices = (
                self.get_polling_table_indices(df_list, must_have, any_of)
                if table_indices is None
                else table_indices
            )
            return self.get_combined_table(
                df_list, table_indices, verbose=True, table_years=table_years
            )

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to get tables: %s", e)
            return None

    @staticmethod
    def _get_pattern_columns(
        df: pd.DataFrame, pattern: str, exclude: str = "net"
    ) -> pd.Index:
        """Get columns matching pattern, excluding specified substring."""
        return pd.Index(
            c
            for c in df.columns
            if pattern.lower() in c.lower() and exclude.lower() not in c.lower()
        )

    def check_completeness(
        self, pattern: str, lower: int, upper: int, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Check completeness of polling data in columns selected for matching pattern.

        Logs warnings about partial/incomplete data but keeps all data."""

        columns = self._get_pattern_columns(df, pattern)
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

    def normalise_poll_data(
        self, df: pd.DataFrame, pattern: str, lower: int, upper: int
    ) -> pd.DataFrame:
        """Normalise polling data by ensuring all columns matching the pattern sum to 100%.

        For rows where ALL columns have data:
        - If sum is within [lower, upper]: normalize to 100%
        - If sum is outside [lower, upper]: set values to NaN (unreliable data)

        Rows with any missing values are skipped. This is different from distribute_undecideds
        which works with partial data."""

        columns = self._get_pattern_columns(df, pattern)
        if len(columns) == 0:
            logger.warning("No columns found matching pattern '%s'", pattern)
            return df

        row_sums = df[columns].sum(axis=1, skipna=True)

        # Only operate on rows where ALL columns have data (no NaN values)
        all_columns_have_data = df[columns].notna().all(axis=1)

        # Check if sum is within or outside acceptable range
        within_range = (row_sums >= lower) & (row_sums <= upper)

        # Normalize rows that are within range and have all data
        to_normalize = all_columns_have_data & within_range
        if to_normalize.any():
            logger.info(
                "Normalising %d rows to 100%% for pattern '%s'",
                to_normalize.sum(),
                pattern,
            )
            for col in columns:
                # Convert to float64 to avoid int-to-float cast errors (pandas 3.0+)
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = df[col].astype("Float64")
                df.loc[to_normalize, col] = (
                    df.loc[to_normalize, col] / row_sums.loc[to_normalize] * 100
                )

        # NaN out rows that are outside range (unreliable data)
        to_nan = all_columns_have_data & ~within_range
        if to_nan.any():
            logger.warning(
                "Setting %d rows to NaN - sum outside [%d, %d] for pattern '%s'",
                to_nan.sum(),
                lower,
                upper,
                pattern,
            )
            for col in columns:
                df.loc[to_nan, col] = pd.NA

        return df.copy()

    def _preprocess_table(
        self,
        raw_df: pd.DataFrame,
        numeric_col_filter: Callable[[str], bool],
    ) -> pd.DataFrame:
        """Common preprocessing for scraped tables.

        Args:
            raw_df: Raw DataFrame from table extraction
            numeric_col_filter: Function that takes column name and returns True if
                               the column should be converted to numeric
        """
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        # Handle inconsistent column names
        if "Brand" not in raw_df.columns:
            for alt in ("Firm", "Polling Firm"):
                if alt in raw_df.columns:
                    raw_df = raw_df.rename(columns={alt: "Brand"})
                    break

        # Delete information/header rows
        header_values = {"Polling Firm", "Firm", "Brand", "Date"}
        mask = raw_df["Brand"].notna() & ~raw_df["Brand"].isin(header_values)
        if "Interview mode" in raw_df.columns:
            mask = mask & (raw_df["Brand"] != raw_df["Interview mode"])
        df = raw_df[mask].copy()

        def _is_text_column(series: pd.Series) -> bool:
            """Check if a column contains text (string/object dtype in pandas 3.0+)."""
            return pd.api.types.is_string_dtype(series) or series.dtype == "object"

        # Strip square bracket footnotes from every string/object column
        for col in df.columns:
            if _is_text_column(df[col]):
                # Convert to string Series for .str accessor (preserves NaN)
                as_str = df[col].astype("string")
                df[col] = as_str.str.replace(r"\[[a-z0-9]+\]", "", regex=True).str.strip()

        # Remove percents and convert to numeric for specified columns
        for col in df.columns:
            if numeric_col_filter(col) and _is_text_column(df[col]):
                as_str = df[col].astype("string")
                df[col] = pd.to_numeric(
                    as_str.str.replace("%", "", regex=False).str.strip(),
                    errors="coerce",
                )

        # Parse dates
        df["parsed_date"] = DateParser.parse_series(df["Date"])

        return df.reset_index(drop=True)

    def flag_problematic_polls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag polls with incomplete data OR that don't sum to expected range.

        - Complete data outside sum range: NaN the values AND flag problematic
        - Incomplete data: Keep partial values AND flag problematic

        Treats missing IND (independents) values as 0 before checking completeness."""

        df = df.copy()

        def check_and_nan_bad_sums(pattern: str, lower: int, upper: int) -> pd.Series:
            cols = [c for c in df.columns if pattern in c.lower()]
            if not cols:
                return pd.Series(False, index=df.index)

            # Identify optional columns (not reported by all pollsters)
            df_check = df[cols].copy()
            opt = [
                c for c in cols
                if "ind" in c.lower() or "oth" in c.lower() or "onp" in c.lower()
            ]

            # Check for any data BEFORE filling optional cols with 0
            has_any_data = df[cols].notna().any(axis=1)

            # Fill optional columns with 0 for sum/completeness checks
            if opt:
                df_check[opt] = df_check[opt].fillna(0)
            all_present = df_check.notna().all(axis=1)
            total = df_check.sum(axis=1)
            outside_range = (total < lower) | (total > upper)

            # Complete but outside range: NaN the values
            complete_bad_sum = all_present & outside_range
            if complete_bad_sum.any():
                df.loc[complete_bad_sum, cols] = np.nan

            # Problematic if: (has data but incomplete) OR (complete but outside range)
            incomplete = has_any_data & ~all_present
            return incomplete | complete_bad_sum

        primary_problem = check_and_nan_bad_sums("primary", self.PRIMARY_VOTE_LOWER, self.PRIMARY_VOTE_UPPER)
        tpp_problem = check_and_nan_bad_sums("2pp", self.TPP_VOTE_LOWER, self.TPP_VOTE_UPPER)

        # Only flag problems on rows that have the relevant data.
        # Alternative 2PP rows (ALP vs ONP) have primary cleared and no L/NP 2PP.
        primary_cols = [c for c in df.columns if "primary" in c.lower()]
        has_primary = df[primary_cols].notna().any(axis=1) if primary_cols else pd.Series(False, index=df.index)
        has_classic_2pp = (
            df.get("2PP vote ALP", pd.Series(dtype="float64")).notna()
            & df.get("2PP vote L/NP", pd.Series(dtype="float64")).notna()
        )
        df["problematic"] = (primary_problem & has_primary) | (tpp_problem & has_classic_2pp)
        return df

    def scrape_voting_intention_polls(
        self, table_indices: list[int] | None = None
    ) -> pd.DataFrame:
        """Scrape polling data using the simple table extraction approach."""
        raw_extracted_df = self.get_tables_simple(table_indices)

        if raw_extracted_df is None or raw_extracted_df.empty:
            logger.warning("No data extracted from tables")
            return pd.DataFrame()

        # Preprocess: rename columns, filter rows, strip footnotes, convert numeric, parse dates
        processed_df = self._preprocess_table(
            raw_extracted_df,
            numeric_col_filter=lambda c: "vote" in c.lower() or "size" in c.lower(),
        )

        if processed_df.empty:
            return pd.DataFrame()

        # Handle duplicate rows from Wikipedia's split 2PP format.
        # Primary votes are only valid on rows with classic ALP vs L/NP 2PP
        # (avoids double-counting from duplicate rows for alternative 2PP matchups).
        # All 2PP columns are kept regardless.
        alp_col = "2PP vote ALP"
        lnp_col = "2PP vote L/NP"
        primary_cols = [c for c in processed_df.columns if "primary" in c.lower()]
        if alp_col in processed_df.columns and lnp_col in processed_df.columns and primary_cols:
            tpp_cols = [c for c in processed_df.columns if "2pp" in c.lower()]
            has_any_2pp = processed_df[tpp_cols].notna().any(axis=1)
            has_classic_2pp = processed_df[alp_col].notna() & processed_df[lnp_col].notna()
            # Only clear primary on rows that have SOME 2PP but not the classic pair
            # (these are alternative 2PP rows like ALP vs ONP)
            no_classic_2pp = has_any_2pp & ~has_classic_2pp
            n_cleared = no_classic_2pp.sum()
            if n_cleared:
                processed_df.loc[no_classic_2pp, primary_cols] = np.nan
                logger.info(
                    "Cleared primary vote data from %d rows without classic ALP vs L/NP 2PP",
                    n_cleared,
                )

        # Check for completeness
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
        processed_df = self.normalise_poll_data(
            df=processed_df,
            pattern="primary",
            lower=self.PRIMARY_VOTE_LOWER,
            upper=self.PRIMARY_VOTE_UPPER,
        )
        processed_df = self.normalise_poll_data(
            df=processed_df,
            pattern="2pp",
            lower=self.TPP_VOTE_LOWER,
            upper=self.TPP_VOTE_UPPER,
        )

        # - flag problematic polls that still don't sum to 100
        processed_df = self.flag_problematic_polls(df=processed_df)

        return processed_df

    def scrape_attitudinal_polls(
        self, table_indices: list[int] | None = None
    ) -> pd.DataFrame:
        """Scrape attitudinal and leadership polling data using the simple table extraction approach."""
        # Get Party leaders table (preferred PM)
        pm_df = self.get_tables_simple(table_indices, must_have=["party leaders"])

        # Get Satisfaction table (Albanese/Ley satisfaction)
        sat_df = self.get_tables_simple(table_indices, any_of=["albanese", "ley"])
        if sat_df is not None:
            sat_df = self.fix_satisfaction_columns(sat_df)

        # Merge the tables on Date and Brand/Firm
        if pm_df is not None and sat_df is not None and not sat_df.empty:
            # Ensure both have Brand column
            for df in [pm_df, sat_df]:
                if "Firm" in df.columns and "Brand" not in df.columns:
                    df.rename(columns={"Firm": "Brand"}, inplace=True)
            raw_extracted_df = pd.merge(
                pm_df, sat_df, on=["Date", "Brand", "Sample size"], how="outer"
            )
        elif pm_df is not None:
            raw_extracted_df = pm_df
        elif sat_df is not None:
            raw_extracted_df = sat_df
        else:
            # Fall back to legacy format
            raw_extracted_df = self.get_tables_simple(
                table_indices, must_have=["prime minister"]
            )

        if raw_extracted_df is None or raw_extracted_df.empty:
            logger.warning("No attitudinal polling data extracted from tables")
            return pd.DataFrame()

        # Rename "Party leaders" columns to "Preferred prime minister" for backward compatibility
        col_renames = {
            c: c.replace("Party leaders", "Preferred prime minister")
            for c in raw_extracted_df.columns
            if "Party leaders" in c
        }
        if col_renames:
            raw_extracted_df = raw_extracted_df.rename(columns=col_renames)

        # Define filter for attitudinal columns
        def is_attitudinal_col(col: str) -> bool:
            col_lower = col.lower()
            return (
                "ley" in col_lower
                or "albanese" in col_lower
                or "prime minister" in col_lower
                or "party leaders" in col_lower
            )

        # Preprocess: rename columns, filter rows, strip footnotes, convert numeric, parse dates
        processed_df = self._preprocess_table(
            raw_extracted_df,
            numeric_col_filter=is_attitudinal_col,
        )

        if processed_df.empty:
            return pd.DataFrame()

        # Define the groups with specific patterns for completeness checking
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
            processed_df = self.normalise_poll_data(
                df=processed_df,
                pattern=group,
                lower=self.ATTITUDINAL_VOTE_LOWER,
                upper=self.ATTITUDINAL_VOTE_UPPER,
            )

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

    def _save_df(
        self, df: pd.DataFrame, prefix: str, today: str, data_dir: Path
    ) -> None:
        """Save a DataFrame to CSV if not empty."""
        if df.empty:
            return
        filepath = data_dir / f"{prefix}_{self.election_year}_{today}.csv"
        df.to_csv(filepath, index=False)
        logger.info("Saved %s data to %s", prefix, filepath)

    def save_data(self, vi_df: pd.DataFrame, pm_df: pd.DataFrame) -> None:
        """Save polling data to CSV files with today's date in filename."""
        data_dir = Path("../poll-data")
        data_dir.mkdir(exist_ok=True)
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        self._save_df(vi_df, "voting_intention", today, data_dir)
        self._save_df(pm_df, "preferred_pm", today, data_dir)


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
                        vi_df["parsed_date"].dropna().min().strftime("%Y-%m-%d"),
                        vi_df["parsed_date"].dropna().max().strftime("%Y-%m-%d"),
                    )
                    logger.info("Polling firms: %d", vi_df["Brand"].nunique())

                if not pm_df.empty:
                    logger.info("Attitudinal - Total polls: %d", len(pm_df))
                    logger.info(
                        "Date range: %s to %s",
                        pm_df["parsed_date"].dropna().min().strftime("%Y-%m-%d"),
                        pm_df["parsed_date"].dropna().max().strftime("%Y-%m-%d"),
                    )
            else:
                logger.error("No polling data found for %s", election)

    except Exception as e:
        logger.error("Script failed: %s", e)
        raise


if __name__ == "__main__":
    main()
