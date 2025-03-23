import os
from praw import Reddit
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT', 'f1rstaid:v1.0')
}

SUBREDDITS = [
    'f1visa',
    'optcpt',
    'immigration',
    'internationalstudents'
]

SEARCH_TERMS = [
    'Day 1 CPT',
    'OPT STEM extension',
    'F1 visa renewal',
    'CPT internship',
    'F1 visa interview',
    'OPT unemployment',
    'STEM OPT',
    'F1 transfer',
    'F1 grace period'
]