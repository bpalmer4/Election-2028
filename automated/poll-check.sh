#!/bin/bash
# Daily poll check wrapper script
# 58 6,12,18 * * * /Users/bryanpalmer/Australian-Federal-Election-2028/automated/poll-check.sh   

# set-up parameters
home=/Users/bryanpalmer
project=Australian-Federal-Election-2028
working=automated
runrun=robust_scraper.py

# move to the home directory
cd $home

# activate the uv environment
source $home/$project/.venv/bin/activate

# move to the working directory
cd $project/$working

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

python3 check_new_polls.py >>poll-check-log.log 2>>poll-check-err.log