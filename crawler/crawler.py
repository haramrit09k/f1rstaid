import logging
import json
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# ---------------------------
# CONFIGURATION
# ---------------------------

# Configure logging to output to both a file and the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("crawler.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# List of base URLs you want to crawl.
base_urls = [
    "https://iso.mit.edu",    # Example site with wp-admin rules.
    "https://www.cmu.edu/oie"   # Example site with custom disallowed paths.
]

# Keywords for F-1 student relevance.
f1_keywords = [
    "CPT", "Curricular Practical Training", "OPT", "Optional Practical Training", "international student", "foreign student", "F-1 visa", "F1 visa", "student visa", "international tax", "international tax return", "social security tax", "medicare tax", "employment tax", "tax filing", "tax obligations", "tax return", "tax guide", "taxes for students", "taxes for international students", "taxes for foreign students", "taxes for F-1 students", "taxes for F1 students", "taxes for foreign scholars", "taxes for foreign researchers", "taxes for foreign exchange visitors"
]

# Files for output and checkpointing.
ROBOTS_MAPPING_FILE = "websites_robots.json"   # Mapping of base URL to robots.txt (single-line).
RELEVANT_URLS_FILE = "relevant_urls.txt"         # Final list of relevant URLs.
CHECKPOINT_FILE = "crawler_state.json"           # Checkpoint file for pause/resume.

# Set to True to resume from a previous checkpoint if available.
RESUME = True

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def get_robots_txt(base_url):
    """
    Download the robots.txt with retries and proxy handling
    """
    robots_url = base_url.rstrip("/") + "/robots.txt"
    session = requests.Session()
    
    # Configure retries
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.get(
            robots_url,
            timeout=10,
            proxies={'http': None, 'https': None},  # Bypass proxies
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        if response.status_code == 200:
            single_line = " ".join(response.text.split())
            logger.info(f"Fetched robots.txt from {robots_url}")
            return single_line
        else:
            logger.warning(f"Received status code {response.status_code} for {robots_url}")
            return ""
    except Exception as e:
        logger.error(f"Error fetching {robots_url}: {e}")
        return ""

def parse_robots_txt(robots_txt):
    """
    Parse robots.txt content to extract disallowed and allowed paths.
    Returns two lists: disallowed_paths and allowed_paths.
    Note: This is a very basic parser.
    """
    disallowed = []
    allowed = []
    if not robots_txt:
        return disallowed, allowed
    tokens = robots_txt.split()
    for i, token in enumerate(tokens):
        if token.lower().startswith("disallow:"):
            path = token[9:].strip() if len(token) > 9 else (tokens[i+1] if i+1 < len(tokens) else "")
            if path:
                disallowed.append(path)
        elif token.lower().startswith("allow:"):
            path = token[6:].strip() if len(token) > 6 else (tokens[i+1] if i+1 < len(tokens) else "")
            if path:
                allowed.append(path)
    return disallowed, allowed

def is_allowed(url, disallowed, allowed):
    """
    Determine if a URL is allowed by comparing its path against the disallowed and allowed lists.
    Allowed paths override disallowed paths.
    """
    parsed = urlparse(url)
    path = parsed.path
    for allow_path in allowed:
        if allow_path != "" and path.startswith(allow_path):
            return True
    for dis_path in disallowed:
        if dis_path != "" and path.startswith(dis_path):
            logger.debug(f"URL {url} is disallowed because it starts with {dis_path}")
            return False
    return True

def extract_main_content(soup):
    """
    Remove common boilerplate elements from the soup and return the main text.
    This function removes elements like <nav>, <header>, <footer>, and <aside>.
    """
    # Remove boilerplate elements.
    for tag in soup(["nav", "header", "footer", "aside"]):
        tag.decompose()
    
    # Optionally, if there's a main content tag, focus on that.
    main = soup.find("main")
    if main:
        text = main.get_text(separator=" ", strip=True)
    else:
        # Fallback: return text from entire page.
        text = soup.get_text(separator=" ", strip=True)
    return text

def is_relevant(content):
    """
    Determine if a page is narrowly relevant for F-1 topics (e.g., CPT, OPT)
    using a weighted scoring system. This version expects that 'content' is
    the main text of the page (with boilerplate removed).
    """
    # Define positive keywords with weights.
    positive_keywords = {
        "cpt": 5,
        "opt": 5,
        "curricular practical training": 4,
        "optional practical training": 4,
        "work authorization": 3,
        "employment authorization": 3,
        "internship": 2,
        "practical training": 3
    }
    # Negative keywords to avoid false positives from boilerplate or generic content.
    negative_keywords = {
        "international student": 2,
        "immigration": 2,
        "global affairs": 2,
        "study abroad": 2,
        "university homepage": 2,
        "contact us": 2
    }
    
    text = content.lower()
    score = 0
    
    # Count positive occurrences.
    for keyword, weight in positive_keywords.items():
        occurrences = text.count(keyword)
        score += weight * occurrences
        
    # Subtract points for negative occurrences.
    for keyword, weight in negative_keywords.items():
        occurrences = text.count(keyword)
        score -= weight * occurrences
    logging.debug(f"Relevance score: {score}")
    # Define a threshold that you can adjust based on experimentation.
    return score >= 20

def save_checkpoint(state):
    """Save the crawler state to a checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Checkpoint saved with {len(state['visited'])} visited URLs, {len(state['to_visit'])} URLs to visit, and {len(state['relevant_urls'])} relevant URLs.")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")

def load_checkpoint():
    """Load the crawler state from a checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            state = json.load(f)
        logger.info(f"Resumed from checkpoint: {len(state['visited'])} visited URLs, {len(state['to_visit'])} URLs to visit, and {len(state['relevant_urls'])} relevant URLs.")
        return state
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None

# ---------------------------
# STEP 1: Build Robots.txt Mapping for All Base URLs
# ---------------------------
robots_mapping = {}

for base_url in base_urls:
    robots_txt = get_robots_txt(base_url)
    robots_mapping[base_url] = robots_txt

with open(ROBOTS_MAPPING_FILE, "w") as f:
    json.dump(robots_mapping, f, indent=2)
logger.info(f"Saved robots.txt mapping to {ROBOTS_MAPPING_FILE}")

# ---------------------------
# STEP 2: Crawl Websites and Filter for F-1 Relevant URLs with Checkpointing
# ---------------------------

# Overall state structure
state = {
    "visited": {},
    "to_visit": {},
    "relevant_urls": {}
}

# For each base URL, we'll store separate lists in our state.
for base_url in base_urls:
    state["visited"][base_url] = []
    state["to_visit"][base_url] = [base_url]
    state["relevant_urls"][base_url] = []

# If RESUME is True and a checkpoint exists, load it.
if RESUME and os.path.exists(CHECKPOINT_FILE):
    loaded_state = load_checkpoint()
    if loaded_state:
        state = loaded_state

# Process each website separately.
try:
    for base_url in base_urls:
        logger.info(f"Starting crawl for {base_url}")
        domain = urlparse(base_url).netloc
        
        # Create a persistent session with retries
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Parse robots.txt for this site
        disallowed, allowed = parse_robots_txt(robots_mapping.get(base_url, ""))
        logger.info(f"For {base_url}: Disallowed paths: {disallowed} | Allowed paths: {allowed}")

        # Load state for this base_url if available
        visited = set(state["visited"].get(base_url, []))
        to_visit = state["to_visit"].get(base_url, [base_url])
        relevant_urls = set(state["relevant_urls"].get(base_url, []))
        batch_counter = 0  # Counter to trigger batch saving

        while to_visit:
            url = to_visit.pop(0)
            if url in visited:
                logger.debug(f"Already visited {url}, skipping.")
                continue
            if not is_allowed(url, disallowed, allowed):
                logger.info(f"Skipping disallowed URL: {url}")
                continue

            visited.add(url)
            try:
                response = session.get(
                    url,
                    timeout=15,
                    proxies={'http': None, 'https': None},  # Explicitly disable proxies
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept-Language': 'en-US,en;q=0.9'
                    }
                )
                # PythonAnywhere whitelist check
                if "pythonanywhere" in response.text.lower():
                    logger.error("Blocked by PythonAnywhere's whitelist")
                    raise Exception("Domain not whitelisted on PythonAnywhere")
                
                if response.status_code != 200:
                    logger.warning(f"Non-200 status code {response.status_code} for {url}")
                    continue
            except Exception as e:
                logger.error(f"Error accessing {url}: {e}")
                time.sleep(5)  # Add delay between retries
                continue
            
            page_text = response.text
            soup = BeautifulSoup(page_text, "html.parser")
            # Extract the main content to avoid boilerplate text.
            main_content = extract_main_content(soup)
            if is_relevant(main_content):
                logger.info(f"Relevant page found: {url}")
                if url not in relevant_urls:
                    relevant_urls.add(url)
                    batch_counter += 1

            soup = BeautifulSoup(page_text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                link = urljoin(url, a_tag['href'])
                parsed_link = urlparse(link)
                # Follow only valid HTTP links within the same domain.
                if parsed_link.scheme in ("http", "https") and domain in parsed_link.netloc:
                    if not is_allowed(link, disallowed, allowed):
                        logger.debug(f"Link {link} on {url} is disallowed, skipping.")
                        continue
                    if link not in visited:
                        to_visit.append(link)

            # Update state for this base_url.
            state["visited"][base_url] = list(visited)
            state["to_visit"][base_url] = to_visit
            state["relevant_urls"][base_url] = list(relevant_urls)

            # Save checkpoint every 10 new relevant URLs.
            if batch_counter >= 10:
                save_checkpoint(state)
                batch_counter = 0

        logger.info(f"Finished crawling {base_url}. Visited {len(visited)} pages; found {len(relevant_urls)} relevant pages.")
        # Update state in case we finished this base_url.
        state["visited"][base_url] = list(visited)
        state["to_visit"][base_url] = to_visit
        state["relevant_urls"][base_url] = list(relevant_urls)
        save_checkpoint(state)

except KeyboardInterrupt:
    logger.info("Pause requested by user. Saving current state...")
    save_checkpoint(state)
    logger.info("Exiting gracefully due to KeyboardInterrupt.")

# ---------------------------
# STEP 3: Save Relevant URLs to a Final File for ML Ingestion
# ---------------------------
all_relevant = []
for base_url in base_urls:
    all_relevant.extend(state["relevant_urls"].get(base_url, []))

with open(RELEVANT_URLS_FILE, "w") as f:
    for url in all_relevant:
        f.write(url + "\n")
logger.info(f"Saved total of {len(all_relevant)} relevant URLs to {RELEVANT_URLS_FILE}")
