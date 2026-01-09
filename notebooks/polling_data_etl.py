
from pathlib import Path
import pandas as pd

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
            raise FileNotFoundError(f"No {data_type} data files found. Please run the scraper first.")

    # Load the data
    df = pd.read_csv(data_path)
    if "parsed_date" in df.columns:
        df.index = pd.PeriodIndex(df["parsed_date"], freq="D")

    df = df.dropna(axis=1, how="all")  # drop all NAN columns

    # Filter out alternative 2PP rows for voting intention data
    # Keep only classic ALP vs L/NP rows, exclude ALP vs ONP rows
    if data_type == "voting_intention":
        tpp_lnp_col = next(
            (c for c in df.columns if "2pp" in c.lower() and "l/np" in c.lower()),
            None,
        )
        tpp_onp_col = next(
            (c for c in df.columns if "2pp" in c.lower() and "onp" in c.lower()),
            None,
        )
        if tpp_lnp_col and tpp_onp_col:
            alt_tpp_rows = df[tpp_lnp_col].isna() & df[tpp_onp_col].notna()
            if alt_tpp_rows.any():
                print(f"Filtering out {alt_tpp_rows.sum()} alternative 2PP rows (ALP vs ONP)")
                df = df[~alt_tpp_rows].copy()

    return df