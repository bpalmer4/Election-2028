#!/usr/bin/env python3
"""
Scrape Australian federal election polling data from Wikipedia.

This script scrapes polling data from Wikipedia pages for opinion polling
on Australian federal elections, focusing on voting intention and 
preferred Prime Minister data.
"""

import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaPollingScaper:
    """Scraper for Australian federal election polling data from Wikipedia."""
    
    def __init__(self, election_year: str = "next"):
        if election_year == "next":
            self.url = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Australian_federal_election"
        else:
            self.url = f"https://en.wikipedia.org/wiki/Opinion_polling_for_the_{election_year}_Australian_federal_election"
        
        self.election_year = election_year
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def fetch_page(self) -> BeautifulSoup:
        """Fetch and parse the Wikipedia page."""
        try:
            response = self.session.get(self.url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            logger.info(f"Successfully fetched Wikipedia page for {self.election_year}")
            return soup
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Wikipedia page: {e}")
            raise
    
    def parse_date_range(self, date_str: str) -> Optional[datetime]:
        """
        Parse date ranges and return the midpoint date.
        Handles formats like:
        - "13–18 Jul 2025" -> midpoint between 13 Jul and 18 Jul
        - "Jul 13–18, 2025" -> midpoint between Jul 13 and Jul 18
        - "13 Jul 2025" -> exact date
        """
        if not date_str or date_str.strip() == '':
            return None
            
        # Clean up the date string
        date_str = date_str.strip().replace('\u00a0', ' ')
        
        # Handle various date range formats
        range_patterns = [
            # "13–18 Jul 2025" or "13-18 Jul 2025" 
            r'(\d{1,2})\s*[–\-—]\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})',
            # "Jul 13–18, 2025" or "July 13-18, 2025"
            r'([A-Za-z]+)\s+(\d{1,2})\s*[–\-—]\s*(\d{1,2}),?\s+(\d{4})',
            # "13 Jul–18 Aug 2025" (cross-month ranges)
            r'(\d{1,2})\s+([A-Za-z]+)\s*[–\-—]\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})',
            # "28 Apr – 2 May 2025" (cross-month with spaces)
            r'(\d{1,2})\s+([A-Za-z]+)\s*[–\-—]\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})',
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, date_str)
            if match:
                groups = match.groups()
                
                if len(groups) == 4:
                    if groups[0].isdigit():  # Pattern 1: "13–18 Jul 2025"
                        start_day, end_day, month, year = groups
                        start_date = self.parse_single_date(f"{start_day} {month} {year}")
                        end_date = self.parse_single_date(f"{end_day} {month} {year}")
                    else:  # Pattern 2: "Jul 13–18, 2025"
                        month, start_day, end_day, year = groups
                        start_date = self.parse_single_date(f"{start_day} {month} {year}")
                        end_date = self.parse_single_date(f"{end_day} {month} {year}")
                        
                elif len(groups) == 5:  # Pattern 3: "13 Jul–18 Aug 2025"
                    start_day, start_month, end_day, end_month, year = groups
                    start_date = self.parse_single_date(f"{start_day} {start_month} {year}")
                    end_date = self.parse_single_date(f"{end_day} {end_month} {year}")
                
                if start_date and end_date:
                    # Calculate midpoint
                    midpoint = start_date + (end_date - start_date) / 2
                    logger.debug(f"Parsed date range '{date_str}' -> midpoint: {midpoint.strftime('%Y-%m-%d')}")
                    return midpoint
        
        # If no range pattern matched, try parsing as single date
        single_date = self.parse_single_date(date_str)
        if single_date:
            logger.debug(f"Parsed single date '{date_str}' -> {single_date.strftime('%Y-%m-%d')}")
            return single_date
        
        logger.warning(f"Could not parse date: '{date_str}'")
        return None
    
    def parse_single_date(self, date_str: str) -> Optional[datetime]:
        """Parse a single date in various formats."""
        if not date_str:
            return None
        
        # Handle approximate date formats like "Late June 2025"
        approx_patterns = [
            (r'late\s+(\w+)\s+(\d{4})', lambda m: self._parse_late_month(m.group(1), int(m.group(2)))),
            (r'early\s+(\w+)\s+(\d{4})', lambda m: self._parse_early_month(m.group(1), int(m.group(2)))),
            (r'mid\s+(\w+)\s+(\d{4})', lambda m: self._parse_mid_month(m.group(1), int(m.group(2)))),
        ]
        
        for pattern, parser in approx_patterns:
            match = re.search(pattern, date_str.lower())
            if match:
                try:
                    return parser(match)
                except:
                    continue
            
        # Common date formats
        date_formats = [
            '%d %b %Y',      # 18 Jul 2025
            '%d %B %Y',      # 18 July 2025
            '%b %d, %Y',     # Jul 18, 2025
            '%B %d, %Y',     # July 18, 2025
            '%Y-%m-%d',      # 2025-07-18
            '%d/%m/%Y',      # 18/07/2025
            '%d %b',         # 18 Jul (assume current year)
            '%b %d',         # Jul 18 (assume current year)
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # If no year specified, assume current year
                if parsed_date.year == 1900:
                    parsed_date = parsed_date.replace(year=datetime.now().year)
                return parsed_date
            except ValueError:
                continue
                
        return None
    
    def _parse_late_month(self, month_name: str, year: int) -> datetime:
        """Parse 'Late Month Year' to last week of the month."""
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month_num = month_map.get(month_name.lower())
        if month_num:
            # Use 25th as approximation for "late month"
            return datetime(year, month_num, 25)
        raise ValueError(f"Unknown month: {month_name}")
    
    def _parse_early_month(self, month_name: str, year: int) -> datetime:
        """Parse 'Early Month Year' to first week of the month."""
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month_num = month_map.get(month_name.lower())
        if month_num:
            # Use 5th as approximation for "early month"
            return datetime(year, month_num, 5)
        raise ValueError(f"Unknown month: {month_name}")
    
    def _parse_mid_month(self, month_name: str, year: int) -> datetime:
        """Parse 'Mid Month Year' to middle of the month."""
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month_num = month_map.get(month_name.lower())
        if month_num:
            # Use 15th as approximation for "mid month"
            return datetime(year, month_num, 15)
        raise ValueError(f"Unknown month: {month_name}")
    
    def extract_percentage(self, text: str) -> Optional[float]:
        """Extract percentage value from text."""
        if not text:
            return None
            
        # Remove non-breaking spaces and clean up
        text = text.replace('\u00a0', ' ').strip()
        
        # Look for percentage pattern
        match = re.search(r'(\d+(?:\.\d+)?)%?', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
                
        return None
    
    def identify_table_type(self, table: Tag) -> Optional[str]:
        """Identify if table contains voting intention or preferred PM data."""
        # Look at table headers and nearby headings
        table_text = table.get_text().lower()
        
        # Check preceding headings
        prev_elements = []
        current = table
        for _ in range(5):  # Look back 5 elements
            current = current.find_previous_sibling()
            if current is None:
                break
            if current.name in ['h2', 'h3', 'h4']:
                prev_elements.append(current.get_text().lower())
        
        combined_text = ' '.join(prev_elements) + ' ' + table_text
        
        if 'preferred prime minister' in combined_text or 'preferred pm' in combined_text:
            return 'preferred_pm'
        elif 'voting intention' in combined_text or 'primary vote' in combined_text:
            return 'voting_intention'
        elif any(party in table_text for party in ['alp', 'labor', 'liberal', 'coalition']):
            # Default to voting intention if it has party data
            return 'voting_intention'
        
        return None
    
    def find_polling_tables(self, soup: BeautifulSoup) -> List[Tuple[Tag, str]]:
        """Find the main national voting intention table."""
        tables = []
        
        # Look for the first voting intention table which should be the national one
        for table in soup.find_all('table', class_='wikitable'):
            table_type = self.identify_table_type(table)
            if table_type == 'voting_intention':
                # Check if this table has the full column structure we expect
                try:
                    df_list = pd.read_html(str(table), header=[0, 1])
                    if df_list:
                        df = df_list[0]
                        # Check for key columns that indicate this is the main national table
                        all_text = str(df.columns).lower()
                        if 'primary vote' in all_text and 'alp' in all_text and '2pp' in all_text:
                            # Additional check: ensure this is the main national table by checking preceding headings
                            prev_headings = []
                            current = table
                            for _ in range(3):
                                current = current.find_previous_sibling()
                                if current is None:
                                    break
                                if current.name in ['h2', 'h3']:
                                    prev_headings.append(current.get_text().lower())
                            
                            heading_text = ' '.join(prev_headings)
                            # Skip state-specific or non-national tables
                            if any(state in heading_text for state in ['queensland', 'nsw', 'victoria', 'western australia', 'south australia', 'tasmania', 'act', 'nt', 'by state', 'state-by-state']):
                                logger.debug(f"Skipping state-specific table under heading: {heading_text}")
                                continue
                            
                            logger.info(f"Found main national voting intention table with heading context: {heading_text}")
                            tables.append((table, table_type))
                            break  # Only take the first/main table
                except Exception as e:
                    logger.debug(f"Error processing table: {e}")
                    continue
                    
        logger.info(f"Selected {len(tables)} main polling table")
        return tables
    
    def extract_table_data(self, table: Tag, table_type: str) -> List[Dict]:
        """Extract polling data using pandas to handle multiindex columns."""
        try:
            # Convert HTML table to pandas DataFrame with multiindex columns
            df_list = pd.read_html(str(table), header=[0, 1])
            if not df_list:
                logger.warning("No tables found by pandas")
                return []
            
            df = df_list[0]  # Take the first table
            logger.debug(f"Pandas found table with shape: {df.shape}")
            
            # Handle multiindex columns by flattening them
            if isinstance(df.columns, pd.MultiIndex):
                # Combine multiindex levels, removing empty parts
                new_columns = []
                for col in df.columns:
                    # Join non-empty parts of the column name
                    parts = [str(part).strip() for part in col if str(part).strip() and str(part) != 'Unnamed']
                    combined = ' '.join(parts)
                    # Remove footnote references
                    combined = re.sub(r'\[[a-z0-9]+\]', '', combined).strip()
                    new_columns.append(combined)
                df.columns = new_columns
            else:
                # Clean single-level columns
                df.columns = [re.sub(r'\[[a-z0-9]+\]', '', str(col)).strip() for col in df.columns]
            
            logger.debug(f"Cleaned columns: {list(df.columns)}")
            
            # Look specifically for UND/Undecided patterns
            und_columns = [col for col in df.columns if 'und' in col.lower() or 'undecided' in col.lower()]
            logger.info(f"UND-related columns found: {und_columns}")
            
            # Debug: show all columns for first table with UND
            if und_columns:
                logger.info(f"All columns in table with UND: {list(df.columns)}")
            
            # Filter out informational rows (rows without valid dates)
            rows = []
            for idx, row in df.iterrows():
                try:
                    # Check if first column looks like a date
                    date_text = str(row.iloc[0]).strip()
                    if not date_text or date_text.lower() in ['nan', 'none', '']:
                        continue
                    
                    # Try to parse the date
                    date_parsed = self.parse_date_range(date_text)
                    if date_parsed is None:
                        continue
                    
                    # Build row data
                    row_data = {
                        'original_date': date_text,  # Keep original date string
                        'date': date_parsed,         # Parsed/standardized date
                        'table_type': table_type,
                        'source': f'wikipedia_{self.election_year}'
                    }
                    
                    # Map columns to data (flexible matching)
                    for col_name in df.columns:
                        col_lower = col_name.lower()
                        cell_value = str(row[col_name]).strip()
                        
                        # Skip empty/nan values
                        if not cell_value or cell_value.lower() in ['nan', 'none', '']:
                            continue
                        
                        # Map to appropriate fields
                        if 'brand' in col_lower:
                            # Remove footnote references from brand/firm names
                            clean_brand = re.sub(r'\[[a-z0-9]+\]', '', cell_value).strip()
                            row_data['brand'] = clean_brand
                        elif 'interview' in col_lower and 'mode' in col_lower:
                            # Remove footnote references from interview mode
                            clean_mode = re.sub(r'\[[a-z0-9]+\]', '', cell_value).strip()
                            row_data['interview_mode'] = clean_mode
                        elif 'sample' in col_lower:
                            # Remove footnote references from sample size  
                            clean_sample = re.sub(r'\[[a-z0-9]+\]', '', cell_value).strip()
                            row_data['sample_size'] = clean_sample
                        elif 'primary' in col_lower and 'alp' in col_lower:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['primary_alp'] = pct
                        elif 'primary' in col_lower and ('l/np' in col_lower or 'lnp' in col_lower):
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['primary_lnp'] = pct
                        elif 'primary' in col_lower and 'grn' in col_lower:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['primary_grn'] = pct
                        elif 'primary' in col_lower and 'onp' in col_lower:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['primary_onp'] = pct
                        elif 'primary' in col_lower and 'ind' in col_lower:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['primary_ind'] = pct
                        elif 'primary' in col_lower and 'oth' in col_lower:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['primary_oth'] = pct
                        elif ('primary' in col_lower and 'und' in col_lower) or col_lower == 'primary vote und':
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['primary_und'] = pct
                        elif '2pp' in col_lower and 'alp' in col_lower:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['2pp_alp'] = pct
                        elif '2pp' in col_lower and ('l/np' in col_lower or 'lnp' in col_lower):
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                row_data['2pp_lnp'] = pct
                        # Simple party name columns (for when party names are individual columns)
                        elif col_lower in ['alp', 'labor'] and len(row_data) > 3:  # Assume it's vote data
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                if '2pp_alp' not in row_data:  # Prefer explicit 2pp columns
                                    row_data['vote_alp'] = pct
                        elif col_lower in ['l/np', 'lnp', 'coalition'] and len(row_data) > 3:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                if '2pp_lnp' not in row_data:
                                    row_data['vote_lnp'] = pct
                        elif col_lower == 'grn' and len(row_data) > 3:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                if 'primary_grn' not in row_data:
                                    row_data['primary_grn'] = pct
                        elif col_lower == 'onp' and len(row_data) > 3:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                if 'primary_onp' not in row_data:
                                    row_data['primary_onp'] = pct
                        elif col_lower == 'ind' and len(row_data) > 3:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                if 'primary_ind' not in row_data:
                                    row_data['primary_ind'] = pct
                        elif col_lower == 'oth' and len(row_data) > 3:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                if 'primary_oth' not in row_data:
                                    row_data['primary_oth'] = pct
                        elif (col_lower == 'und' or col_lower == 'undecided') and len(row_data) > 3:
                            pct = self.extract_percentage(cell_value)
                            if pct is not None:
                                if 'primary_und' not in row_data:
                                    row_data['primary_und'] = pct
                    
                    # Skip election results (not polling data)
                    if 'brand' in row_data and row_data['brand'].lower() in ['election', 'election result']:
                        logger.debug(f"Skipping election result row: {row_data.get('date', 'no date')}")
                        continue
                    
                    # Only keep rows with meaningful polling data
                    polling_fields = ['primary_alp', 'primary_lnp', 'primary_grn', 'primary_onp', 
                                    'primary_ind', 'primary_oth', 'primary_und', '2pp_alp', '2pp_lnp', 
                                    'vote_alp', 'vote_lnp']
                    if any(field in row_data for field in polling_fields):
                        rows.append(row_data)
                
                except Exception as e:
                    logger.debug(f"Skipping row {idx}: {e}")
                    continue
            
            logger.info(f"Extracted {len(rows)} polling entries from {table_type} table using pandas")
            return rows
            
        except Exception as e:
            logger.error(f"Failed to extract table data with pandas: {e}")
            return []
    
    def scrape_polls(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scrape all polling data and return separate DataFrames for each type."""
        soup = self.fetch_page()
        tables = self.find_polling_tables(soup)
        
        voting_intention_data = []
        preferred_pm_data = []
        
        for table, table_type in tables:
            table_data = self.extract_table_data(table, table_type)
            
            if table_type == 'voting_intention':
                voting_intention_data.extend(table_data)
            elif table_type == 'preferred_pm':
                preferred_pm_data.extend(table_data)
        
        # Create DataFrames
        vi_df = pd.DataFrame(voting_intention_data) if voting_intention_data else pd.DataFrame()
        pm_df = pd.DataFrame(preferred_pm_data) if preferred_pm_data else pd.DataFrame()
        
        # Clean and sort data
        for df in [vi_df, pm_df]:
            if not df.empty:
                # Only use columns that exist for deduplication
                dedup_cols = ['date']
                if 'brand' in df.columns:
                    dedup_cols.append('brand')
                elif 'polling_firm' in df.columns:
                    dedup_cols.append('polling_firm')
                
                df.drop_duplicates(subset=dedup_cols, inplace=True)
                df.sort_values('date', inplace=True)
                df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Successfully scraped {len(vi_df)} voting intention and {len(pm_df)} preferred PM entries")
        
        return vi_df, pm_df
    
    def save_data(self, vi_df: pd.DataFrame, pm_df: pd.DataFrame) -> None:
        """Save polling data to CSV files with today's date in filename."""
        # Ensure poll-data directory exists
        data_dir = Path("../poll-data")
        data_dir.mkdir(exist_ok=True)
        
        # Get today's date for filename
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Save voting intention data
        if not vi_df.empty:
            vi_df_to_save = vi_df.copy()
            vi_df_to_save['date'] = vi_df_to_save['date'].dt.strftime('%Y-%m-%d')
            
            filename = f"voting_intention_{self.election_year}_{today}.csv"
            filepath = data_dir / filename
            vi_df_to_save.to_csv(filepath, index=False)
            logger.info(f"Saved voting intention data to {filepath}")
        
        # Save preferred PM data
        if not pm_df.empty:
            pm_df_to_save = pm_df.copy()
            pm_df_to_save['date'] = pm_df_to_save['date'].dt.strftime('%Y-%m-%d')
            
            filename = f"preferred_pm_{self.election_year}_{today}.csv"
            filepath = data_dir / filename
            pm_df_to_save.to_csv(filepath, index=False)
            logger.info(f"Saved preferred PM data to {filepath}")

def main():
    """Main execution function."""
    try:
        # Scrape only next election data
        elections = ["next"]
        
        for election in elections:
            print(f"\n=== Scraping {election} election data ===")
            scraper = WikipediaPollingScaper(election)
            print(f"URL: {scraper.url}")
            vi_df, pm_df = scraper.scrape_polls()
            
            if not vi_df.empty or not pm_df.empty:
                scraper.save_data(vi_df, pm_df)
                
                # Print summary
                if not vi_df.empty:
                    print(f"Voting Intention - Total polls: {len(vi_df)}")
                    print(f"Date range: {vi_df['date'].min().strftime('%Y-%m-%d')} to {vi_df['date'].max().strftime('%Y-%m-%d')}")
                    print(f"Polling firms: {vi_df['brand'].nunique()}")
                
                if not pm_df.empty:
                    print(f"Preferred PM - Total polls: {len(pm_df)}")
                    print(f"Date range: {pm_df['date'].min().strftime('%Y-%m-%d')} to {pm_df['date'].max().strftime('%Y-%m-%d')}")
            else:
                print(f"No polling data found for {election}")
            
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise

if __name__ == "__main__":
    main()