"""Previous election results for annotations."""

from functools import cache

import pandas as pd

COALITION_LIST = ["LP", "LNP", "NP", "CLP"]
OTHER_LIST = ["DLP", "DEM", "UAP", "Others"]
COALITION = "Coalition"
OTHER = "Other"


@cache
def get_last_election() -> pd.Series:
    """Retrieve the latest election results for annotations."""

    # Capture latest value for 2025 election annotations
    election_df = pd.read_csv(
        "../historic-data/election-outcomes.csv",
        sep=" ",
        skipinitialspace=True,
        header=0,
        comment="#",
        na_values=["", "-"],
    ).fillna(0)  # make sure NA values are 0 for numeric addition

    # Convert all columns to numeric where possible
    for col in election_df.columns:
        try:
            election_df[col] = pd.to_numeric(election_df[col])
        except ValueError:
            continue

    # only keep numeric columns and filter for 2025 election results
    df_numeric = election_df.select_dtypes(include=["number"])
    election_2025 = df_numeric[df_numeric["Year"] == 2025].iloc[0]

    # Calculate Coalition and Others totals
    election_2025[COALITION] = election_2025[COALITION_LIST].sum()
    election_2025[OTHER] = election_2025[OTHER_LIST].sum()

    # Clean up to keep only relevant fields
    election_2025.drop(labels=["Year"] + COALITION_LIST + OTHER_LIST, inplace=True)

    return election_2025


def get_election_result(column: str) -> float | None:
    """Get the election result for a specific poll column.

    Args:
        column: Poll column name (e.g., "2PP vote ALP", "Primary vote L/NP")

    Returns:
        Election result percentage, or None if not found
    """
    election = get_last_election()
    results_map = {
        "2PP vote ALP": election["Labor2pp"],
        "2PP vote L/NP": 100 - election["Labor2pp"],
        "Primary vote ALP": election["ALP"],
        "Primary vote L/NP": election[COALITION],
        "Primary vote GRN": election["GRN"],
        "Primary vote ONP": election["ON"],
        "Others Primary Vote": election[OTHER],
    }
    return results_map.get(column)
