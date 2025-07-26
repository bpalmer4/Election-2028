#!/usr/bin/env python3
"""Daily check for new polls on Wikipedia - sends email if new polls found."""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add notebooks directory to path
sys.path.append(str(Path(__file__).parent.parent / "notebooks"))

from scrape_wikipedia_polls import WikipediaPollingScaper
from notifications import send_success_notification, send_error_notification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_yesterday_poll_count() -> tuple[int, int]:
    """Get poll counts from yesterday's files."""
    poll_dir = Path(__file__).parent.parent / "poll-data"
    
    # Find most recent files
    vi_files = sorted(poll_dir.glob("voting_intention_*.csv"))
    pm_files = sorted(poll_dir.glob("preferred_pm_*.csv"))
    
    vi_count = pm_count = 0
    
    if vi_files:
        import pandas as pd
        df = pd.read_csv(vi_files[-1])
        vi_count = len(df)
    
    if pm_files:
        import pandas as pd
        df = pd.read_csv(pm_files[-1])
        pm_count = len(df)
    
    return vi_count, pm_count


def main():
    try:
        # Get yesterday's counts
        old_vi_count, old_pm_count = get_yesterday_poll_count()
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
            message = f"🗳️ NEW POLLS FOUND!\n\n"
            if new_vi > 0:
                message += f"• {new_vi} new voting intention polls\n"
            if new_pm > 0:
                message += f"• {new_pm} new preferred PM polls\n"
            message += f"\nTotal: {new_vi_count} VI polls, {new_pm_count} PM polls"
            
            logger.info("Sending notification...")
            send_success_notification(message)
        else:
            logger.info("No new polls found")
            
    except Exception as e:
        logger.error(f"Poll check failed: {e}")
        send_error_notification(str(e))


if __name__ == "__main__":
    main()