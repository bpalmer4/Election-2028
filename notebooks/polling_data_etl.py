from pathlib import Path

import numpy as np
import pandas as pd

# 2pp pair sums must fall in this range post-renormalisation; otherwise the
# pair is cleared.
TPP_SUM_LOWER = 99
TPP_SUM_UPPER = 101

# Essential publishes 2pp with implicit undecided; the two reported values
# sum below 100. If the raw sum is in this band we redistribute proportionally
# to total 100.
ESSENTIAL_REDIST_LOWER = 85
ESSENTIAL_REDIST_UPPER = 99


def _validate_and_normalise_2pp(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise 2pp values on every row.

    Each Wikipedia row reports one 2pp matchup, so any row with any
    ``2PP vote ...`` value should have exactly two such values populated.
    Violations are cleared.

    For Essential rows where the two populated 2pp values sum to
    [ESSENTIAL_REDIST_LOWER, ESSENTIAL_REDIST_UPPER] (implicit undecided),
    the values are renormalised proportionally to 100.

    Any 2pp pair whose post-renormalisation sum still falls outside
    [TPP_SUM_LOWER, TPP_SUM_UPPER] is cleared.
    """
    tpp_cols = [c for c in df.columns if c.startswith("2PP vote ")]
    if len(tpp_cols) < 2 or "Brand" not in df.columns:
        return df

    df = df.copy()

    # 1. Each row with 2pp data should have exactly two non-NaN 2pp values.
    n_populated = df[tpp_cols].notna().sum(axis=1)
    bad_shape = (n_populated > 0) & (n_populated != 2)
    if bad_shape.any():
        print(
            f"⚠️  Clearing 2pp on {bad_shape.sum()} row(s) with != 2 populated "
            f"2pp values."
        )
        df.loc[bad_shape, tpp_cols] = np.nan

    # 2. Essential implicit-undecided redistribution (matchup-agnostic).
    has_pair = df[tpp_cols].notna().sum(axis=1) == 2
    is_essential = df["Brand"].str.contains("Essential", case=False, na=False)
    sum_2pp = df[tpp_cols].sum(axis=1, skipna=True)
    needs_redist = (
        has_pair
        & is_essential
        & sum_2pp.between(ESSENTIAL_REDIST_LOWER, ESSENTIAL_REDIST_UPPER)
    )
    if needs_redist.any():
        print(
            f"Redistributing implicit undecided in {needs_redist.sum()} "
            f"Essential 2pp row(s)."
        )
        for col in tpp_cols:
            df.loc[needs_redist, col] = (
                df.loc[needs_redist, col] / sum_2pp[needs_redist] * 100
            )

    # 3. Post-redistribution sum check — clear pairs outside [99, 101].
    sum_2pp = df[tpp_cols].sum(axis=1, skipna=True)
    has_pair = df[tpp_cols].notna().sum(axis=1) == 2
    bad_sum = has_pair & ~sum_2pp.between(TPP_SUM_LOWER, TPP_SUM_UPPER)
    if bad_sum.any():
        print(
            f"⚠️  Clearing 2pp on {bad_sum.sum()} row(s) whose pair sums "
            f"outside [{TPP_SUM_LOWER}, {TPP_SUM_UPPER}]."
        )
        df.loc[bad_sum, tpp_cols] = np.nan

    return df


def load_polling_data(data_type: str = "voting_intention") -> pd.DataFrame:
    """Load the most recent polling data with data freshness validation.

    Args:
        data_type: Either "voting_intention" or "preferred_pm"
    """
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    data_dir = Path("../poll-data")

    # Set file prefix based on data type
    file_prefix = data_type

    # Look for today's file first
    today_file = data_dir / f"{file_prefix}_next_{today}.csv"

    if today_file.exists():
        data_path = today_file
        print(f"Using today's {data_type} data file: {data_path}")
    else:
        # Look for any recent file with date suffix
        pattern = f"{file_prefix}_next_*.csv"
        recent_files = list(data_dir.glob(pattern))

        help_msg = (
            "Consider running the scraper first: automated/scrape_wikipedia_polls.py"
        )
        if recent_files:
            # Sort by filename (date suffix) and take most recent
            most_recent = sorted(recent_files)[-1]
            file_date = most_recent.stem.split("_")[-1]  # Extract date from filename

            print(f"⚠️  WARNING: No {data_type} data file found for today ({today})")
            print(f"Using most recent file: {most_recent} (from {file_date})")
            print(help_msg)
            data_path = most_recent
        else:
            raise FileNotFoundError(
                f"No {data_type} data files found. Please run the scraper first."
            )

    # Load the data
    df = pd.read_csv(data_path)
    if "parsed_date" in df.columns:
        df.index = pd.PeriodIndex(df["parsed_date"], freq="D")

    df = df.dropna(axis=1, how="all")  # drop all NAN columns

    # Note: rows in the CSV represent (poll, 2pp-matchup) pairs, not polls.
    # Wikipedia splits a single poll across multiple rows when more than one
    # 2pp matchup is reported (classic ALP-vs-L/NP, ALP-vs-ONP, etc.). The
    # scraper NaN's primary votes on non-classic rows so per-column NaN
    # filtering downstream is sufficient to de-duplicate primaries.
    if data_type == "voting_intention":
        df = _validate_and_normalise_2pp(df)

    return df
