import os
from dotenv import load_dotenv
import streamlit as st  # Only needed for st.secrets; safe to import locally

# Load environment variables from .env file
load_dotenv()

def get_secret(group, key, env_var=None):
    """
    Retrieve a secret from st.secrets first, then fall back to os.getenv.
    
    Parameters:
        group (str): The group name in the secrets TOML (e.g., "reddit").
        key (str): The key within that group (e.g., "client_id").
        env_var (str): Optional; the environment variable name to fall back on.
                        Defaults to GROUP_KEY in uppercase.
    Returns:
        The secret value as a string, or None if not found.
    """
    if env_var is None:
        env_var = f"{group.upper()}_{key.upper()}"
    
    # Attempt to get from st.secrets if available.
    try:
        value = st.secrets[group][key]
        if value:
            return value
    except Exception:
        pass  # st.secrets might not be available locally
    
    # Fallback to environment variable.
    return os.getenv(env_var)

# Use the helper to load your Reddit credentials.
REDDIT_CONFIG = {
    'client_id': get_secret("reddit", "client_id"),
    'client_secret': get_secret("reddit", "client_secret"),
    'user_agent': get_secret("reddit", "user_agent") or 'f1rstaid:v1.0'
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