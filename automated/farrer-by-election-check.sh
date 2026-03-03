#!/bin/zsh

# set-up parameters
home=/Users/bryanpalmer
project=Election-2028
working=automated
runrun=farrer_scraper.py

# move to the home directory
cd $home

# activate the uv environment
source $home/$project/.venv/bin/activate

# move to the working directory
cd $project/$working

# initiate the data capture with Farrer by-election scraper
python ./$runrun >>z-farrer-log.log 2>>z-farrer-err.log

# Only commit and push if scraper succeeded (exit code 0)
if [ $? -eq 0 ]; then
    cd $home/$project
    git add "betting-data/sportsbet-2026-farrer-by-election.csv"
    git commit -m "farrer by-election betting update"
    git push
    echo "$(date): Scraper succeeded, data committed" >>automated/z-farrer-log.log
else
    echo "$(date): Scraper failed, no commit made" >>automated/z-farrer-err.log
fi
