# Election-2028

Polling and betting analysis for the next Australian Federal Election, which must be held before the middle of 2028.

## Overview

This project provides automated data capture, analysis, and visualization of:
- **Betting odds** from Sportsbet for election winner markets
- **Polling data** from Wikipedia (voting intention and preferred PM)
- **Statistical analysis** using LOESS smoothing and Bayesian methods (PyMC)

## Features

- Daily automated scraping of Sportsbet betting odds
- Three-times-daily polling data capture from Wikipedia
- LOESS trend analysis with 90-day smoothing
- Bayesian Generalized Random Walk (GRW) model for poll aggregation
- House effects estimation for pollsters
- Email notifications for data changes and scraper failures
- Historical election context (1946-2025)

## Directory Structure

```
Election-2028/
├── notebooks/           # Jupyter notebooks for analysis & development
│   ├── pymc/            # PyMC Bayesian analysis modules
│   ├── common.py        # Shared constants, party colors, utilities
│   ├── polling_data_etl.py
│   └── *.ipynb          # Analysis notebooks
├── automated/           # Production scripts & cron jobs
│   ├── robust_scraper.py        # Sportsbet odds scraper
│   ├── check_new_polls.py       # Poll monitoring
│   ├── scrape_wikipedia_polls.py
│   └── notifications.py
├── betting-data/        # Captured betting odds (CSV)
├── poll-data/           # Polling data snapshots (CSV)
├── charts/              # Generated visualizations
│   ├── election-winner/
│   ├── LOESS-trend/
│   ├── bayesian-aggregation/
│   └── previous-elections/
└── historic-data/       # Historical election results (1946-2025)
```

## Requirements

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) package manager
- Chrome/Chromium (for Selenium-based scraping)

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd Election-2028

# Install dependencies with uv
uv sync

# Install custom plotting library
./test-mgplot.sh

# Configure email notifications (optional)
cp automated/.env.example automated/.env
# Edit .env with your SMTP credentials
```

## Usage

### Manual Data Capture

```bash
# Capture betting odds
cd automated/
python robust_scraper.py

# Check for new polls
python check_new_polls.py

# Scrape Wikipedia polling data
python scrape_wikipedia_polls.py
```

### Running Notebooks

```bash
# Start Jupyter
uv run jupyter lab

# Key notebooks:
# - chart_election_winner_at_sportsbet.ipynb  (betting visualizations)
# - chart_LOESS_Trend_Analysis.ipynb          (polling trends)
# - bayesian_grw_voting_intention.ipynb       (Bayesian aggregation)
```

### Code Quality

```bash
# Lint Python files and notebooks
./notebooks/lint-all.sh notebooks/
```

## Automation (Cron Jobs)

The project includes cron job scripts for automated data capture:

| Script | Schedule | Purpose |
|--------|----------|---------|
| `betting-market-check.sh` | 6:32 AM daily | Capture Sportsbet odds |
| `poll-check.sh` | 6:58 AM, 12:58 PM, 6:58 PM | Check for new polls |

## Data Formats

### Betting Data
```csv
Datetime,Party,Odds
2025-05-07 16:22:45,Labor,1.33
2025-05-07 16:22:45,Coalition,3.35
```

### Polling Data (Voting Intention)
```csv
Date,Brand,Sample size,Primary vote ALP,Primary vote L/NP,2PP vote ALP,...
```

### Polling Data (Preferred PM)
```csv
Date,Brand,Sample size,Preferred prime minister Albanese,Preferred prime minister Ley,...
```

## Key Dependencies

- **Data Science**: pandas, numpy, matplotlib, statsmodels
- **Web Scraping**: selenium, beautifulsoup4, webdriver_manager
- **Bayesian**: pymc, arviz, jax, numpyro
- **Visualization**: mgplot (custom), matplotlib
- **Utilities**: readabs, python-decouple

## License

Private project - all rights reserved.
