#!/bin/bash
# Scrape Wikipedia polling data
# Deletes any existing files from today before scraping

# set-up parameters
home=/Users/bryanpalmer
project=Election-2028

# move to the project directory
cd "$home/$project"

# activate the uv environment
source "$home/$project/.venv/bin/activate"

# get today's date
today=$(date +%Y-%m-%d)

# delete any existing poll files from today
echo "Removing any existing poll files from $today..."
rm -f poll-data/*_${today}.csv

# run the scraper
echo "Scraping Wikipedia polls..."
cd automated
python scrape_wikipedia_polls.py

echo "Done."
