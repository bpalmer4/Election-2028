#!/bin/bash
# Daily poll check wrapper script
# 58 6,12,18 * * * /Users/bryanpalmer/Election-2028/automated/poll-check.sh   

# set-up parameters
home=/Users/bryanpalmer
project=Election-2028
working=automated

# move to the home directory
cd $home

# activate the uv environment
source $home/$project/.venv/bin/activate

# move to the working directory
cd $project/$working
python check_new_polls.py >>poll-check-log.log 2>>poll-check-err.log
