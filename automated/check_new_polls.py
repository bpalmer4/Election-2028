"""Daily check for new polls on Wikipedia - sends email if new polls found."""

import logging
from pathlib import Path

import pandas as pd
from scrape_wikipedia_polls import WikipediaPollingScaper
from notifications import send_success_notification, send_error_notification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_poll_changes(
    new_df: pd.DataFrame, old_df: pd.DataFrame
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Get (pollster, date) tuples for new and updated polls."""
    if old_df.empty:
        new_items = [
            (row["Brand"], str(row["parsed_date"])) for _, row in new_df.iterrows()
        ]
        return new_items, []

    key_cols = ["Brand", "parsed_date"]

    # Ensure parsed_date is string in both dataframes for comparison
    new_df = new_df.copy()
    old_df = old_df.copy()
    new_df["parsed_date"] = new_df["parsed_date"].astype(str)
    old_df["parsed_date"] = old_df["parsed_date"].astype(str)

    # New polls: keys in new_df but not in old_df
    old_keys = set(old_df[key_cols].apply(tuple, 1))  # type: ignore[call-overload]
    new_keys = set(new_df[key_cols].apply(tuple, 1))  # type: ignore[call-overload]
    truly_new_keys = new_keys - old_keys
    new_polls = new_df[new_df[key_cols].apply(tuple, 1).isin(truly_new_keys)]  # type: ignore[call-overload]
    new_items = [(row["Brand"], row["parsed_date"]) for _, row in new_polls.iterrows()]

    # Updated polls: same key exists in both, but row data differs
    common_keys = new_keys & old_keys

    # Only compare columns that aren't all lowercase (ignore metadata columns we add)
    # But keep key columns for filtering, and exclude Date (raw text, constantly reformatted)
    compare_cols = [
        col
        for col in new_df.columns
        if (not col.islower() or col in key_cols) and col != "Date"
    ]
    common_compare_cols = [col for col in compare_cols if col in old_df.columns]

    # Compare only the data columns
    new_compare = new_df[common_compare_cols]
    old_compare = old_df[common_compare_cols]

    all_changes = pd.concat([new_compare, old_compare]).drop_duplicates(keep=False)
    # Filter to only keys that exist in both dataframes
    changed_with_common_keys = all_changes[
        all_changes[key_cols].apply(tuple, 1).isin(common_keys)  # type: ignore[call-overload]
    ]
    # Get unique keys (avoid showing duplicates from both dataframes)
    updated_keys = set(changed_with_common_keys[key_cols].apply(tuple, 1))  # type: ignore[call-overload]
    updated_items = list(updated_keys)

    return new_items, updated_items


def get_previous_poll_data():
    """Get poll dataframes from previous files (excluding today's)."""
    from datetime import date

    poll_dir = Path(__file__).parent.parent / "poll-data"
    today = date.today().strftime("%Y-%m-%d")

    # Find most recent files excluding today's
    vi_files = sorted(
        [f for f in poll_dir.glob("voting_intention_*.csv") if today not in f.name]
    )
    pm_files = sorted(
        [f for f in poll_dir.glob("preferred_pm_*.csv") if today not in f.name]
    )

    old_vi_df = pd.DataFrame()
    old_pm_df = pd.DataFrame()

    if vi_files:
        old_vi_df = pd.read_csv(vi_files[-1])

    if pm_files:
        old_pm_df = pd.read_csv(pm_files[-1])

    return old_vi_df, old_pm_df


def main():
    try:
        # Get previous poll data
        old_vi_df, old_pm_df = get_previous_poll_data()
        old_vi_count, old_pm_count = len(old_vi_df), len(old_pm_df)
        logger.info(f"Previous counts: VI={old_vi_count}, PM={old_pm_count}")

        # Get today's polls
        scraper = WikipediaPollingScaper("next")
        vi_df, pm_df = scraper.scrape_polls()

        new_vi_count = len(vi_df)
        new_pm_count = len(pm_df)
        logger.info(f"Current counts: VI={new_vi_count}, PM={new_pm_count}")

        # Check for new polls
        new_vi = new_vi_count - old_vi_count
        new_pm = new_pm_count - old_pm_count

        if new_vi > 0 or new_pm > 0:
            # Save new data
            scraper.save_data(vi_df, pm_df)

            # Send notification
            message = "🗳️ NEW POLLS FOUND!\n\n"

            for poll_type, new_df, old_df, count in [
                ("voting intention", vi_df, old_vi_df, new_vi),
                ("preferred PM", pm_df, old_pm_df, new_pm),
            ]:
                if count > 0:
                    new_items, updated_items = get_poll_changes(new_df, old_df)
                    if new_items:
                        message += f"• {len(new_items)} new {poll_type} polls\n"
                        for brand, date in new_items:
                            message += f"  {brand} ({date})\n"
                    if updated_items:
                        message += f"• {len(updated_items)} updated {poll_type} polls\n"
                        for brand, date in updated_items:
                            message += f"  {brand} ({date})\n"

            message += f"\nTotal: {new_vi_count} VI polls, {new_pm_count} PM polls"

            logger.info("Sending notification...")
            send_success_notification(message, "Wikipedia Polls Scraper")
        else:
            logger.info("No new polls found")

    except Exception as e:
        logger.error(f"Poll check failed: {e}")
        send_error_notification(str(e), scraper_name="Wikipedia Polls Scraper")


if __name__ == "__main__":
    main()
