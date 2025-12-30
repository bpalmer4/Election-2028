"""Shared utilities for automated scripts."""

from pathlib import Path

import pandas as pd


def ensure(condition: bool, message: str = "Assertion failed") -> None:
    """Assert a condition, raising AssertionError if False.

    Unlike assert statements, this cannot be disabled with -O flag.
    """
    if not condition:
        raise AssertionError(message)


def find_latest_csv(directory: Path, pattern: str) -> Path | None:
    """Find the most recent CSV file matching a glob pattern.

    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g., "voting_intention_*.csv")

    Returns:
        Path to most recent file (by filename sort), or None if no matches.
    """
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def load_latest_csv(directory: Path, pattern: str) -> pd.DataFrame:
    """Load the most recent CSV file matching a pattern.

    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g., "voting_intention_*.csv")

    Returns:
        DataFrame from the most recent matching file, or empty DataFrame if none.
    """
    path = find_latest_csv(directory, pattern)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)
