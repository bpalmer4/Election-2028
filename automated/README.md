# Robust Election Scraper

This directory contains an improved version of the election betting scraper with email notifications and better error handling.

## Files

- `robust_scraper.py` - Main scraper with fallback mechanisms
- `config.py` - Configuration settings and CSS selectors
- `notifications.py` - Email notification system  
- `election-winner.sh` - Updated cron job script
- `.env.example` - Example environment configuration

## Setup

### 1. Email Notifications

Copy the example environment file and configure your email settings:

```bash
cp .env.example .env
```

Edit `.env` with your email settings:

```bash
# For Gmail (recommended)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password  # Not your regular password!
RECIPIENT_EMAIL=your-email@gmail.com
```

**Important for Gmail users:**
1. Enable 2-factor authentication on your Google account
2. Generate an app-specific password at: https://myaccount.google.com/apppasswords
3. Use the app password, not your regular Gmail password

### 2. Update Cron Job

The cron script has been updated to use the robust scraper. Make sure it's executable:

```bash
chmod +x election-winner.sh
```

Your existing crontab entry should work:
```
58 6 * * * /Users/bryanpalmer/Australian-Federal-Election-2028/automated/election-winner.sh
```

### 3. Test the Scraper

Test manually before relying on the cron job:

```bash
cd /Users/bryanpalmer/Australian-Federal-Election-2028/automated
python robust_scraper.py
```

## Key Improvements

### Robustness
- **Multiple CSS selectors**: Falls back through different selectors when website changes
- **Retry logic**: Attempts scraping multiple times with delays
- **Better error handling**: Catches and logs specific failure points
- **Graceful degradation**: Continues trying alternative approaches

### Notifications
- **Email alerts**: Get notified immediately when scraping fails
- **Success notifications**: Optional confirmation when scraping works
- **Detailed error info**: Full traceback information in failure emails

### Reliability  
- **Exit codes**: Script exits with proper codes for cron monitoring
- **Conditional commits**: Only commits data to git when scraping succeeds
- **Enhanced logging**: Better visibility into what's working/failing

## Monitoring

- Check `scraper.log` for detailed execution logs
- Check `winner-log.log` and `winner-err.log` for cron job output
- Email notifications provide immediate failure alerts

## Troubleshooting

If scraping fails repeatedly:

1. Check the website manually - layout may have changed significantly
2. Update CSS selectors in `config.py`
3. Check Chrome/ChromeDriver compatibility
4. Verify internet connectivity and website availability

The scraper is designed to be more resilient to minor website changes, but major redesigns may still require manual updates to the CSS selectors.