#!/bin/zsh

# set-up parameters
home=/Users/bryanpalmer
project=Election-2028
working=automated
runrun=next_pm_scraper.py

# move to the home directory
cd $home

# activate the uv environment
source $home/$project/.venv/bin/activate

# move to the working directory
cd $project/$working

# initiate the data capture with next PM scraper
python ./$runrun >>z-next-pm-log.log 2>>z-next-pm-err.log

# Only commit and push if scraper succeeded (exit code 0)
if [ $? -eq 0 ]; then
    cd $home/$project
    git add "betting-data/sportsbet-2028-next-pm.csv"
    git commit -m "next PM betting update"
    git push
    echo "$(date): Scraper succeeded, data committed" >>automated/z-next-pm-log.log
else
    echo "$(date): Scraper failed, no commit made" >>automated/z-next-pm-err.log
fi
