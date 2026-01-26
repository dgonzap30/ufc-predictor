"""
Scraper for upcoming UFC events from UFCStats.com and BestFightOdds.com
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from pathlib import Path


def normalize_fighter_name(name: str) -> str:
    """
    Normalize fighter names for matching between different sources.

    Args:
        name: Fighter name to normalize

    Returns:
        Normalized fighter name (lowercase, stripped, standardized)
    """
    # Lowercase and strip
    normalized = name.lower().strip()

    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)

    # Standardize common variations
    normalized = normalized.replace('jr.', 'jr')
    normalized = normalized.replace('sr.', 'sr')

    # Remove periods
    normalized = normalized.replace('.', '')

    return normalized


def get_event_odds() -> Dict[str, Dict]:
    """
    Scrape BestFightOdds.com to get odds for upcoming UFC events.

    Returns:
        dict: Odds lookup keyed by normalized fighter name:
            {
                "fighter name": {
                    "odds": -150,
                    "opponent": "opponent name"
                },
                ...
            }
        Returns empty dict on failure
    """
    try:
        response = requests.get(
            "https://www.bestfightodds.com/",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        odds_dict = {}

        # Find all odds tables
        odds_tables = soup.find_all('table', class_='odds-table')

        for table in odds_tables:
            rows = table.find_all('tr')

            # Look for consecutive rows with fighter names
            i = 0
            while i < len(rows) - 1:
                row1 = rows[i]
                row2 = rows[i + 1]

                # Check if both rows have fighter links
                fighter1_link = row1.find('a', href=lambda x: x and '/fighters/' in x if x else False)
                fighter2_link = row2.find('a', href=lambda x: x and '/fighters/' in x if x else False)

                if fighter1_link and fighter2_link:
                    fighter1_name = fighter1_link.get_text(strip=True)
                    fighter2_name = fighter2_link.get_text(strip=True)

                    # Get all cells in both rows
                    cells1 = row1.find_all('td')
                    cells2 = row2.find_all('td')

                    # Extract odds from cells (skip first cell which has fighter name)
                    odds1 = None
                    odds2 = None

                    # Try to get odds from the cells
                    for cell in cells1[1:]:  # Skip first cell (fighter name)
                        text = cell.get_text(strip=True)
                        # Remove arrows and extract number
                        match = re.search(r'([+-]\d+)', text)
                        if match:
                            try:
                                odds1 = int(match.group(1))
                                break
                            except ValueError:
                                continue

                    for cell in cells2[1:]:  # Skip first cell (fighter name)
                        text = cell.get_text(strip=True)
                        match = re.search(r'([+-]\d+)', text)
                        if match:
                            try:
                                odds2 = int(match.group(1))
                                break
                            except ValueError:
                                continue

                    # Store in lookup dict
                    if odds1 is not None:
                        odds_dict[normalize_fighter_name(fighter1_name)] = {
                            "odds": odds1,
                            "opponent": fighter2_name
                        }

                    if odds2 is not None:
                        odds_dict[normalize_fighter_name(fighter2_name)] = {
                            "odds": odds2,
                            "opponent": fighter1_name
                        }

                    # Skip the second row since we processed it
                    i += 2
                else:
                    i += 1

        print(f"Loaded odds for {len(odds_dict)} fighters from BestFightOdds")
        return odds_dict

    except requests.RequestException as e:
        print(f"Network error fetching odds from BestFightOdds: {e}")
        return {}
    except Exception as e:
        print(f"Error parsing BestFightOdds: {e}")
        return {}


def get_historical_odds(event_date: str, fighters: List[str]) -> Dict[str, Dict]:
    """
    Look up historical odds from the raw dataset for completed events.

    Args:
        event_date: Date string in format "December 06, 2025"
        fighters: List of fighter names appearing on the card

    Returns:
        dict: Odds lookup keyed by normalized fighter name:
            {
                "fighter name": {
                    "odds": -150,
                    "opponent": "opponent name"
                },
                ...
            }
    """
    try:
        # Load historical odds dataset
        odds_path = Path("data/raw/ufc_betting_odds.csv")
        if not odds_path.exists():
            print("Historical odds dataset not found")
            return {}

        df = pd.read_csv(odds_path, low_memory=False)

        # Convert event_date from "December 06, 2025" to "2025-12-06"
        try:
            dt = datetime.strptime(event_date, "%B %d, %Y")
            target_date = dt.strftime("%Y-%m-%d")
        except ValueError:
            print(f"Could not parse event date: {event_date}")
            return {}

        # Filter to this event's date
        # Check BOTH exact date and next day (timezone differences in dataset)
        event_odds = df[df['event_date'] == target_date]
        next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        event_odds_next = df[df['event_date'] == next_day]

        # Combine both - this handles events that might be split across days in the dataset
        combined_odds = pd.concat([event_odds, event_odds_next])

        if combined_odds.empty:
            print(f"No historical odds found for {target_date} or {next_day}")
            return {}

        print(f"Found {len(event_odds)} odds on {target_date}, {len(event_odds_next)} on {next_day}")
        print(f"Using {len(combined_odds)} total fight odds entries")

        event_odds = combined_odds

        # Build lookup dict matching get_event_odds() format
        odds_dict = {}

        # Helper to convert decimal odds to American odds
        def decimal_to_american(decimal_odds):
            if decimal_odds >= 2.0:
                # Underdog: positive American odds
                return int((decimal_odds - 1) * 100)
            else:
                # Favorite: negative American odds
                return int(-100 / (decimal_odds - 1))

        for _, row in event_odds.iterrows():
            f1_name = row['fighter_1']
            f2_name = row['fighter_2']
            f1_odds_decimal = row['odds_1']
            f2_odds_decimal = row['odds_2']

            # Skip if odds are missing
            if pd.isna(f1_odds_decimal) or pd.isna(f2_odds_decimal):
                continue

            # Convert decimal to American format
            try:
                f1_odds = decimal_to_american(float(f1_odds_decimal))
                f2_odds = decimal_to_american(float(f2_odds_decimal))
            except (ValueError, TypeError, ZeroDivisionError):
                continue

            # Store in lookup dict
            odds_dict[normalize_fighter_name(f1_name)] = {
                "odds": f1_odds,
                "opponent": f2_name
            }
            odds_dict[normalize_fighter_name(f2_name)] = {
                "odds": f2_odds,
                "opponent": f1_name
            }

        print(f"Loaded historical odds for {len(odds_dict)} fighter entries")
        return odds_dict

    except Exception as e:
        print(f"Error loading historical odds: {e}")
        return {}


def get_recent_events(limit: int = 10) -> List[Dict[str, str]]:
    """
    Scrape UFCStats.com to get recent completed events.

    Args:
        limit: Maximum number of events to return

    Returns:
        List of dicts with 'name' and 'url' keys
    """
    try:
        response = requests.get(
            "http://ufcstats.com/statistics/events/completed?page=all",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        events = []
        event_links = soup.find_all('a', href=True)
        for link in event_links:
            href = link.get('href', '')
            if 'event-details' in href:
                event_name = link.get_text(strip=True)
                if event_name and len(events) < limit:
                    events.append({
                        'name': event_name,
                        'url': href.strip()
                    })

        return events

    except requests.RequestException as e:
        print(f"Network error fetching completed events: {e}")
        return []
    except Exception as e:
        print(f"Error parsing completed events: {e}")
        return []


def get_next_event_url() -> str:
    """
    Scrape UFCStats.com to find the URL of the next upcoming UFC event.

    Returns:
        str: URL of the next event detail page, or empty string on failure
    """
    try:
        # UFCStats lists upcoming events on the main events page
        response = requests.get(
            "http://ufcstats.com/statistics/events/upcoming",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find event links (UFCStats uses "event-details" in the URL)
        event_links = soup.find_all('a', href=True)
        for link in event_links:
            href = link.get('href', '')
            if 'event-details' in href:
                return href.strip()

        print("Could not find event detail link")
        return ""

    except requests.RequestException as e:
        print(f"Network error fetching events: {e}")
        return ""
    except Exception as e:
        print(f"Error parsing events page: {e}")
        return ""


def get_fight_card(event_url: str) -> Dict:
    """
    Scrape fight card details from a UFC event page.

    Args:
        event_url: URL of the event detail page

    Returns:
        dict: Event data with structure:
            {
                "event_name": str,
                "date": str,
                "location": str,
                "fights": [
                    {
                        "fighter1": str,
                        "fighter2": str,
                        "weight_class": str,
                        "is_main_event": bool
                    },
                    ...
                ]
            }
        Returns empty dict on failure
    """
    if not event_url:
        return {}

    try:
        response = requests.get(
            event_url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract event name
        event_name_elem = soup.find('h2', class_='b-content__title')
        event_name = event_name_elem.get_text(strip=True) if event_name_elem else "Unknown Event"

        # Extract event details (date, location)
        event_details = soup.find('ul', class_='b-list__box-list')
        date_str = "TBA"
        location_str = "TBA"

        if event_details:
            detail_items = event_details.find_all('li', class_='b-list__box-list-item')
            for item in detail_items:
                text = item.get_text(strip=True)
                if "Date:" in text:
                    date_str = text.replace("Date:", "").strip()
                elif "Location:" in text:
                    location_str = text.replace("Location:", "").strip()

        # Get odds - use historical data for past events, live scraper for upcoming
        try:
            event_dt = datetime.strptime(date_str, "%B %d, %Y")
            is_past_event = event_dt < datetime.now()
        except ValueError:
            # If we can't parse date, assume it's upcoming
            is_past_event = False

        if is_past_event:
            print(f"Fetching historical odds for {date_str}...")
            odds_lookup = get_historical_odds(date_str, [])
        else:
            print("Fetching odds from BestFightOdds...")
            odds_lookup = get_event_odds()

        # Extract fights
        fights = []
        fight_rows = soup.find_all('tr', class_='b-fight-details__table-row')

        fight_count = 0  # Track actual fight count (non-header rows)

        for row in fight_rows:
            # Skip if this is a header row
            if row.get('class') and 'b-fight-details__table-head' in row.get('class'):
                continue

            # Extract fighter names
            fighters = row.find_all('a', class_='b-link_style_black')
            if len(fighters) < 2:
                continue

            fighter1_name = fighters[0].get_text(strip=True)
            fighter2_name = fighters[1].get_text(strip=True)

            # Skip empty rows
            if not fighter1_name or not fighter2_name:
                continue

            # Extract weight class from column 6
            cols = row.find_all('td', class_='b-fight-details__table-col')
            weight_class = "Catchweight Bout"

            if len(cols) > 6:
                weight_class_text = cols[6].get_text(strip=True)
                # Add " Bout" suffix if not already present
                if weight_class_text and "Bout" not in weight_class_text:
                    weight_class = f"{weight_class_text} Bout"
                elif weight_class_text:
                    weight_class = weight_class_text

            # First actual fight is the main event
            is_main_event = (fight_count == 0)
            fight_count += 1

            # Look up odds for both fighters
            f1_normalized = normalize_fighter_name(fighter1_name)
            f2_normalized = normalize_fighter_name(fighter2_name)

            odds_f1 = None
            odds_f2 = None

            # Only use odds if the opponent matches (ensures correct fight)
            if f1_normalized in odds_lookup:
                f1_data = odds_lookup[f1_normalized]
                # Verify opponent matches fighter2
                if normalize_fighter_name(f1_data["opponent"]) == f2_normalized:
                    odds_f1 = f1_data["odds"]

            if f2_normalized in odds_lookup:
                f2_data = odds_lookup[f2_normalized]
                # Verify opponent matches fighter1
                if normalize_fighter_name(f2_data["opponent"]) == f1_normalized:
                    odds_f2 = f2_data["odds"]

            fights.append({
                "fighter1": fighter1_name,
                "fighter2": fighter2_name,
                "weight_class": weight_class,
                "is_main_event": is_main_event,
                "odds_f1": odds_f1,
                "odds_f2": odds_f2
            })

        return {
            "event_name": event_name,
            "date": date_str,
            "location": location_str,
            "fights": fights
        }

    except requests.RequestException as e:
        print(f"Network error fetching event details: {e}")
        return {}
    except Exception as e:
        print(f"Error parsing event page: {e}")
        return {}


# =============================================================================
# PANDAS-BASED ODDS SCRAPING (BestFightOdds.com)
# =============================================================================

def clean_fighter_name(name):
    """
    Cleans messy names like 'Jon JonesJones' or 'Jon Jones\n'
    often found in scraped tables.
    """
    if pd.isna(name):
        return None
    name = str(name).strip()
    return name


def american_to_decimal(odd):
    """
    Converts American Odds (e.g. -200, +150) to Decimal (e.g. 1.5, 2.5).
    """
    try:
        val = float(odd)
        if val == 0:
            return 1.0
        if val > 0:
            return (val / 100) + 1  # +150 => 2.5
        else:
            return (100 / abs(val)) + 1  # -200 => 1.5
    except (ValueError, TypeError):
        return None


def scrape_upcoming_odds():
    """
    Scrapes the 'Future Events' tab from BestFightOdds using pandas.
    Returns a DataFrame with [fighter_name, american_odds, decimal_odds].
    """
    url = "https://www.bestfightodds.com/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        all_odds = []

        for table in tables:
            if len(table.columns) < 3:
                continue

            df = table.copy()
            df.rename(columns={df.columns[0]: 'Fighter'}, inplace=True)

            for _, row in df.iterrows():
                fighter = clean_fighter_name(row['Fighter'])
                if not fighter:
                    continue

                odds_values = []
                for val in row[1:]:
                    try:
                        v = float(val)
                        if abs(v) > 50:
                            odds_values.append(v)
                    except:
                        continue

                if odds_values:
                    best_american = max(odds_values)
                    decimal_odds = american_to_decimal(best_american)

                    if decimal_odds:
                        all_odds.append({
                            'fighter_name': fighter,
                            'american_odds': best_american,
                            'decimal_odds': round(decimal_odds, 3)
                        })

        return pd.DataFrame(all_odds).drop_duplicates(subset=['fighter_name'])

    except Exception as e:
        print(f"Error scraping odds with pandas: {e}")
        return pd.DataFrame()


def get_upcoming_fight_data() -> Dict:
    """
    Master function: Fetches upcoming event and merges with scraped odds.

    Returns:
        dict: Event data with enriched odds information
    """
    # 1. Get event URL and fight card from UFCStats
    event_url = get_next_event_url()
    if not event_url:
        print("Could not find upcoming event URL")
        return {}

    event_data = get_fight_card(event_url)
    if not event_data:
        print("Could not fetch fight card")
        return {}

    # 2. Get odds via pandas scraper
    print("Fetching odds via pandas read_html...")
    odds_df = scrape_upcoming_odds()

    if odds_df.empty:
        print("No odds data found, returning card without odds")
        return event_data

    # 3. Merge odds into fight card
    odds_found = 0
    for fight in event_data.get("fights", []):
        f1_name = fight["fighter1"]
        f2_name = fight["fighter2"]

        # Try to match fighter names in odds dataframe
        for col, fighter_name in [("odds_f1_decimal", f1_name), ("odds_f2_decimal", f2_name)]:
            normalized = normalize_fighter_name(fighter_name)
            match = odds_df[odds_df['fighter_name'].apply(
                lambda x: normalize_fighter_name(str(x)) == normalized
            )]
            if not match.empty:
                key = "odds_f1_decimal" if col == "odds_f1_decimal" else "odds_f2_decimal"
                fight[key] = match.iloc[0]['decimal_odds']
                odds_found += 1

    print(f"Odds found for {odds_found} fighter entries")
    return event_data


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=== Odds Conversion Sanity Check ===")
    test_odds = [-200, 150, -110]
    for o in test_odds:
        print(f"American: {o:+d} -> Decimal: {american_to_decimal(o)}")

    print("\n=== Scraping Upcoming Odds ===")
    df = scrape_upcoming_odds()
    print(f"Found {len(df)} fighters with odds")
    if not df.empty:
        print(df.head(10))

    print("\n=== Full Event Data ===")
    event = get_upcoming_fight_data()
    if event:
        print(f"Event: {event.get('event_name')}")
        print(f"Date: {event.get('date')}")
        for fight in event.get('fights', [])[:3]:
            print(f"  {fight['fighter1']} vs {fight['fighter2']} ({fight['weight_class']})")
