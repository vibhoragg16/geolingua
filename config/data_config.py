# Data collection configuration
REDDIT_SUBREDDITS = {
    'us_south': ['Atlanta', 'Charlotte', 'NewOrleans', 'Austin', 'Birmingham'],
    'uk': ['unitedkingdom', 'CasualUK', 'london', 'manchester'],
    'australia': ['australia', 'melbourne', 'sydney', 'brisbane'],
    'india': ['india', 'bangalore', 'mumbai', 'delhi'],
    'nigeria': ['Nigeria', 'lagos', 'abuja']
}

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


DATA_LIMITS = {
    'reddit_posts': 200,
    'news_articles': 100,
    'wikipedia_articles': 50
}

# Data paths
RAW_DATA_PATH = "/kaggle/input/geolingua-data/data/raw"
PROCESSED_DATA_PATH = "/kaggle/working/data/processed"
DATASETS_PATH = "/kaggle/working/data/datasets"

# Reddit API configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID") or ""
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET") or ""
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT") or ""

# News sources by region
NEWS_SOURCES = {
    'us_south': [
        'https://www.ajc.com',  # Atlanta Journal-Constitution
        'https://www.nola.com',  # New Orleans
        'https://www.charlotteobserver.com',  # Charlotte
        'https://www.statesman.com'  # Austin
    ],
    'uk': [
        'https://www.bbc.co.uk',
        'https://www.theguardian.com',
        'https://www.independent.co.uk'
    ],
    'australia': [
        'https://www.abc.net.au',
        'https://www.smh.com.au',
        'https://www.theage.com.au'
    ],
    'india': [
        'https://www.hindustantimes.com',
        'https://www.thehindu.com',
        'https://indianexpress.com'
    ],
    'nigeria': [
        'https://punchng.com',
        'https://www.vanguardngr.com',
        'https://www.premiumtimesng.com'
    ]
}

# Wikipedia categories by region
WIKIPEDIA_CATEGORIES = {
    'us_south': [
        'Category:Culture of the Southern United States',
        'Category:History of the Southern United States',
        'Category:Southern United States'
    ],
    'uk': [
        'Category:Culture of the United Kingdom',
        'Category:British culture',
        'Category:History of the United Kingdom'
    ],
    'australia': [
        'Category:Culture of Australia',
        'Category:Australian culture',
        'Category:History of Australia'
    ],
    'india': [
        'Category:Culture of India',
        'Category:Indian culture',
        'Category:History of India'
    ],
    'nigeria': [
        'Category:Culture of Nigeria',
        'Category:Nigerian culture',
        'Category:History of Nigeria'
    ]
}

# Data preprocessing configuration
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 1000
LANGUAGE_FILTER = ['en']  # Only English for now
REMOVE_URLS = True
REMOVE_MENTIONS = True
REMOVE_HASHTAGS = False
CLEAN_WHITESPACE = True


# Dataset split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Data quality filters
MIN_SCORE_REDDIT = 5  # Minimum upvotes for Reddit posts
MIN_COMMENT_LENGTH = 20
MAX_COMMENT_LENGTH = 500
FILTER_DELETED = True
FILTER_REMOVED = True

# Geographic markers and indicators
GEOGRAPHIC_MARKERS = {
    'us_south': [
        'y\'all', 'fixin\' to', 'bless your heart', 'sweet tea', 'grits',
        'barbecue', 'hurricane', 'SEC', 'cotton', 'bourbon'
    ],
    'uk': [
        'brilliant', 'cheers', 'bloke', 'quid', 'lorry', 'lift', 'queue',
        'football', 'pub', 'biscuit', 'tea', 'proper'
    ],
    'australia': [
        'mate', 'arvo', 'barbie', 'brekkie', 'fair dinkum', 'footy',
        'servo', 'bottle-o', 'bikkie', 'sunnies'
    ],
    'india': [
        'yaar', 'ji', 'achcha', 'bas', 'chalo', 'jugaad', 'timepass',
        'prepone', 'out of station', 'good name'
    ],
    'nigeria': [
        'abi', 'sha', 'wahala', 'gbam', 'chop', 'wetin', 'dey',
        'small small', 'how far', 'no wahala'
    ]
}