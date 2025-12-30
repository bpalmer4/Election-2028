#!/bin/bash
# Force fresh download from Wikipedia - removes today's files and re-scrapes

set -e

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

TODAY=$(date +%Y-%m-%d)
POLL_DIR="poll-data"

# Remove today's files if they exist
for pattern in "voting_intention_next_${TODAY}.csv" "preferred_pm_next_${TODAY}.csv"; do
    file="${POLL_DIR}/${pattern}"
    if [ -f "$file" ]; then
        echo "Removing: $file"
        rm "$file"
    fi
done

# Run the scraper from automated/ directory (needed for relative imports)
echo "Scraping Wikipedia..."
cd automated
uv run python scrape_wikipedia_polls.py

echo "Done."
