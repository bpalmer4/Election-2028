from functools import lru_cache
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

# 2025 preference flow file used by the 2025-flow derived 2pp builder.
PREF_FLOWS_2025 = Path(
    "../historic-data/preferences/2025-two-party-preferred-flow-by-party.csv"
)

# ALP share of preferences (0–1) for an ALP-vs-ONP matchup. The AEC didn't
# run national ALP-vs-ONP 2pp counts in 2025, so we use two reference flow
# sets. The "Others" bucket combines IND + OTH because pollsters don't all
# break IND out — applying separate IND vs OTH rates would amplify pollster
# bucketing noise rather than reduce it.
#
# Bonham SA 2026 rates — from the Kevin Bonham aggregation post analysing
# the 2026 South Australian election. He published L/NP, GRN and IND only;
# we apply his IND rate to the combined Others bucket.
BONHAM_SA26_FLOWS = {"L/NP": 0.339, "GRN": 0.848, "Others": 0.559}

# Theoretical clean-round flows. GRN strongly prefers ALP over ONP; L/NP
# weakly prefers ONP over ALP; Others sits roughly mid-range (the bucket
# blends ALP-favouring IND with ONP-favouring minor parties).
THEORETICAL_ONP_FLOWS = {"L/NP": 0.25, "GRN": 0.90, "Others": 0.50}


def format_flow_rates(flows: dict[str, float]) -> str:
    """Format an ALP-share flow dict as 'L/NP 33.9, GRN 84.8, Others 55.9'.

    Integers print without a decimal point (25 not 25.0); fractional values
    round to one decimal place.
    """
    parts = []
    for key, val in flows.items():
        pct = round(val * 100, 1)
        if pct == int(pct):
            parts.append(f"{key} {int(pct)}")
        else:
            parts.append(f"{key} {pct}")
    return ", ".join(parts)


@lru_cache(maxsize=1)
def alp_flow_rates_2025() -> dict[str, float]:
    """ALP share (0–1) of preferences for each non-major-party bucket at
    the 2025 federal election (ALP-vs-L/NP context).

    Buckets: GRN, ONP, Others. "Others" is the weighted average of IND +
    every other minor party in the AEC flow file.
    """
    df = pd.read_csv(PREF_FLOWS_2025)
    rates: dict[str, float] = {}
    for ab, key in (("GRN", "GRN"), ("ON", "ONP")):
        row = df[df["PartyAb"] == ab].iloc[0]
        rates[key] = float(row["Australian Labor Party Transfer Percentage"]) / 100.0
    others = df[~df["PartyAb"].isin(["LP", "NP", "GRN", "ON"]) & df["PartyAb"].notna()]
    alp_votes = others["Australian Labor Party Transfer Votes"].sum()
    lnp_votes = others["Liberal/National Coalition Transfer Votes"].sum()
    rates["Others"] = float(alp_votes / (alp_votes + lnp_votes))
    return rates


def derived_2pp_variants() -> list[dict]:
    """Recipes for every derived ALP 2pp series. Each variant produces
    ``2PP {name} ALP`` and ``2PP {name} {counter}`` columns."""
    return [
        {
            "name": "synth-2025",
            "counter": "L/NP",
            "flows": alp_flow_rates_2025(),
            "label": "Synthetic",
        },
        {
            "name": "synth-sa26",
            "counter": "ONP",
            "flows": BONHAM_SA26_FLOWS,
            "label": "SA26 shadow",
        },
        {
            "name": "synth-th",
            "counter": "ONP",
            "flows": THEORETICAL_ONP_FLOWS,
            "label": "Theory shadow",
        },
    ]


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


def _add_others_primary_vote(df: pd.DataFrame) -> pd.DataFrame:
    """Sum non-major primary-vote columns (IND, OTH, etc) into a single
    ``Others Primary Vote`` column. Idempotent.

    Wikipedia pollsters inconsistently break out IND — some lump
    independents into OTH. The combined bucket sidesteps that ambiguity.
    """
    main = ["Primary vote ALP", "Primary vote L/NP", "Primary vote GRN", "Primary vote ONP"]
    others_cols = [c for c in df.columns if "Primary vote" in c and c not in main]
    if not others_cols:
        return df
    df = df.copy()
    df["Others Primary Vote"] = df[others_cols].sum(axis=1, skipna=True)
    all_nan = df[others_cols].isna().all(axis=1)
    df.loc[all_nan, "Others Primary Vote"] = np.nan
    return df


def add_derived_2pp(df: pd.DataFrame, variant: dict) -> pd.DataFrame:
    """Add ``2PP {variant['name']} ALP`` and ``2PP {variant['name']} {counter}``
    columns derived from poll primaries using the variant's flow rates.

    A row gets a derived 2pp only when the four major-party primaries
    (ALP, L/NP, GRN, ONP) are all populated. Buckets not in the flow dict
    contribute nothing to the ALP share — typically ALP itself (100% by
    definition, added directly) and the counter party (0%, omitted).
    """
    df = df.copy()
    required = ["Primary vote ALP", "Primary vote L/NP", "Primary vote GRN", "Primary vote ONP"]
    if not all(c in df.columns for c in required):
        return df
    have_all = df[required].notna().all(axis=1)

    synth_alp = df["Primary vote ALP"].copy()
    for key, rate in variant["flows"].items():
        col = "Others Primary Vote" if key == "Others" else f"Primary vote {key}"
        synth_alp = synth_alp + df[col].fillna(0) * rate

    name = variant["name"]
    counter = variant["counter"]
    df[f"2PP {name} ALP"] = synth_alp.where(have_all)
    df[f"2PP {name} {counter}"] = (100 - synth_alp).where(have_all)
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
        df = _add_others_primary_vote(df)
        for variant in derived_2pp_variants():
            df = add_derived_2pp(df, variant)

    return df
