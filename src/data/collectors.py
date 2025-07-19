import praw
import requests
import time
import json
import os
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
from newspaper import Article
import wikipedia
from urllib.parse import urljoin, urlparse
import re
import feedparser
from bs4 import BeautifulSoup
from bs4 import Tag

from config.data_config import (
    REDDIT_SUBREDDITS, DATA_LIMITS, RAW_DATA_PATH,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    NEWS_SOURCES, WIKIPEDIA_CATEGORIES, MIN_SCORE_REDDIT
)

logger = logging.getLogger(__name__)

class RedditCollector:
    """
    Collector for Reddit data using PRAW (Python Reddit API Wrapper).
    """
    
    def __init__(self, client_id: str = REDDIT_CLIENT_ID, 
                 client_secret: str = REDDIT_CLIENT_SECRET,
                 user_agent: str = REDDIT_USER_AGENT):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit = None
        self.setup_reddit_client()
    
    def setup_reddit_client(self):
        """Initialize Reddit client."""
        try:
            if not self.client_id or not self.client_secret:
                logger.warning("Reddit API credentials not provided. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in data_config.py")
                return
            
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            
            # Test connection
            logger.info(f"Reddit client initialized. Read-only: {self.reddit.read_only}")
            
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            self.reddit = None
    
    def collect_subreddit_posts(self, subreddit_name: str, limit: int = 100, 
                               time_filter: str = 'week') -> List[Dict]:
        """
        Collect posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Number of posts to collect
            time_filter: Time filter ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            List of post data dictionaries
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return []
        
        posts_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts
            hot_posts = subreddit.hot(limit=limit // 2)
            # Get top posts
            top_posts = subreddit.top(time_filter=time_filter, limit=limit // 2)
            
            all_posts = list(hot_posts) + list(top_posts)
            
            for post in all_posts:
                try:
                    # Skip if score is too low
                    if post.score < MIN_SCORE_REDDIT:
                        continue
                    
                    post_data = {
                        'id': post.id,
                        'title': post.title,
                        'selftext': post.selftext,
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'created_utc': post.created_utc,
                        'author': str(post.author) if post.author else '[deleted]',
                        'subreddit': subreddit_name,
                        'url': post.url,
                        'is_self': post.is_self,
                        'stickied': post.stickied,
                        'over_18': post.over_18,
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    posts_data.append(post_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing post {post.id}: {e}")
                    continue
            
            logger.info(f"Collected {len(posts_data)} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error collecting posts from r/{subreddit_name}: {e}")
        
        return posts_data
    
    def collect_post_comments(self, post_id: str, limit: int = 50) -> List[Dict]:
        """
        Collect comments from a specific post.
        
        Args:
            post_id: Reddit post ID
            limit: Number of comments to collect
            
        Returns:
            List of comment data dictionaries
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return []
        
        comments_data = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "more comments" objects
            
            comment_count = 0
            for comment in submission.comments:
                if comment_count >= limit:
                    break
                
                try:
                    # Skip deleted/removed comments
                    if comment.body in ['[deleted]', '[removed]']:
                        continue
                    
                    # Skip comments that are too short
                    if len(comment.body.strip()) < 10:
                        continue
                    
                    comment_data = {
                        'id': comment.id,
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'parent_id': comment.parent_id,
                        'post_id': post_id,
                        'is_submitter': comment.is_submitter,
                        'stickied': comment.stickied,
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    comments_data.append(comment_data)
                    comment_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing comment {comment.id}: {e}")
                    continue
            
            logger.info(f"Collected {len(comments_data)} comments from post {post_id}")
            
        except Exception as e:
            logger.error(f"Error collecting comments from post {post_id}: {e}")
        
        return comments_data
    
    def collect_region_data(self, region: str, limit: int = 200) -> Dict:
        """
        Collect data for a specific region from its associated subreddits.
        
        Args:
            region: Region name (e.g., 'us_south', 'uk', 'australia')
            limit: Total number of posts to collect
            
        Returns:
            Dictionary containing posts and comments data
        """
        if region not in REDDIT_SUBREDDITS:
            logger.error(f"Region {region} not found in REDDIT_SUBREDDITS")
            return {'posts': [], 'comments': []}
        
        subreddits = REDDIT_SUBREDDITS[region]
        posts_per_subreddit = limit // len(subreddits)
        
        all_posts = []
        all_comments = []
        
        for subreddit in subreddits:
            logger.info(f"Collecting from r/{subreddit} for region {region}")
            
            # Collect posts
            posts = self.collect_subreddit_posts(subreddit, posts_per_subreddit)
            all_posts.extend(posts)
            
            # Collect comments from some posts
            for post in posts[:5]:  # Get comments from first 5 posts
                comments = self.collect_post_comments(post['id'], 20)
                all_comments.extend(comments)
            
            time.sleep(1)  # Rate limiting
        
        return {
            'posts': all_posts,
            'comments': all_comments,
            'region': region,
            'collected_at': datetime.now().isoformat()
        }


class NewsCollector:
    """
    Collector for news articles from various sources.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def extract_article_urls(self, source_url: str, max_articles: int = 20) -> list:
        """
        Try to extract article URLs from RSS feed or homepage.
        """
        # Try RSS feed
        rss_candidates = [
            source_url.rstrip('/') + '/rss',
            source_url.rstrip('/') + '/feed',
            source_url.rstrip('/') + '/feeds',
            source_url.rstrip('/') + '/rss.xml',
            source_url.rstrip('/') + '/feed.xml',
        ]
        for rss_url in rss_candidates:
            try:
                feed = feedparser.parse(rss_url)
                if feed.entries:
                    return [entry.link for entry in feed.entries[:max_articles]]
            except Exception:
                continue
        # Fallback: scrape homepage for <a> tags
        try:
            resp = self.session.get(source_url, timeout=10)
            soup = BeautifulSoup(resp.text, features='lxml')
            links = set()
            for a in soup.find_all('a', href=True):
                if isinstance(a, Tag):
                    href = a.get('href')
                    # Some BeautifulSoup versions may return a list for href, ensure string
                    if isinstance(href, list):
                        href = href[0] if href else ''
                    href = str(href)
                    if href.startswith('http') and source_url in href:
                        links.add(href)
                    elif href.startswith('/'):
                        links.add(urljoin(source_url, href))
            # Heuristic: filter out non-article links
            article_links = [l for l in links if len(l) > 30][:max_articles]
            return article_links
        except Exception:
            return []

    def collect_article(self, url: str) -> Optional[Dict]:
        """
        Collect a single news article.
        
        Args:
            url: Article URL
            
        Returns:
            Article data dictionary or None if failed
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Skip if article is too short
            if len(article.text) < 100:
                return None
            
            article_data = {
                'url': url,
                'title': article.title,
                'text': article.text,
                'summary': article.summary,
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'top_image': article.top_image,
                'keywords': article.keywords,
                'collected_at': datetime.now().isoformat()
            }
            
            return article_data
            
        except Exception as e:
            logger.warning(f"Error collecting article from {url}: {e}")
            return None
    
    def collect_from_source(self, source_name: str, urls: list, limit: int = 50) -> list:
        articles = []
        for url in urls[:limit]:
            article = self.collect_article(url)
            if article:
                article['source'] = source_name
                articles.append(article)
            time.sleep(0.5)
        logger.info(f"Collected {len(articles)} articles from {source_name}")
        return articles

    def collect_region_news(self, region: str, limit: int = 100) -> list:
        if region not in NEWS_SOURCES:
            logger.warning(f"No news sources configured for region {region}")
            return []
        all_articles = []
        sources = NEWS_SOURCES[region]
        articles_per_source = max(1, limit // len(sources))
        for url in sources:
            article_urls = self.extract_article_urls(url, articles_per_source)
            articles = self.collect_from_source(url, article_urls, articles_per_source)
            for article in articles:
                article['region'] = region
            all_articles.extend(articles)
        return all_articles


class WikipediaCollector:
    """
    Collector for Wikipedia articles related to specific regions.
    """
    
    def __init__(self):
        wikipedia.set_lang("en")

    def get_articles_from_category(self, category: str, max_articles: int = 10) -> list:
        """
        Use Wikipedia API to get article titles from a category.
        """
        import requests
        S = requests.Session()
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": max_articles,
            "format": "json",
            "cmtype": "page"
        }
        try:
            R = S.get(url=URL, params=PARAMS, timeout=10)
            DATA = R.json()
            pages = DATA.get("query", {}).get("categorymembers", [])
            return [p["title"] for p in pages if p.get("ns") == 0]
        except Exception as e:
            logger.warning(f"Error fetching articles from category {category}: {e}")
            return []

    def collect_article(self, title: str, region: str = None) -> Optional[Dict]:
        """
        Collect a Wikipedia article.
        """
        try:
            page = wikipedia.page(title)
            article_data = {
                'title': page.title,
                'content': page.content,
                'summary': page.summary,
                'url': page.url,
                'categories': page.categories,
                'links': page.links[:50],  # Limit links
                'references': page.references[:20],  # Limit references
                'region': region,
                'collected_at': datetime.now().isoformat()
            }
            return article_data
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Disambiguation: '{title}' is ambiguous, skipping.")
            return None
        except wikipedia.exceptions.PageError:
            logger.warning(f"Page not found: '{title}', skipping.")
            return None
        except Exception as e:
            logger.warning(f"Error collecting Wikipedia article '{title}': {e}")
            return None

    def search_and_collect(self, query: str, region: str, limit: int = 10) -> List[Dict]:
        """
        Search for and collect Wikipedia articles based on a query.
        
        Args:
            query: Search query
            region: Region this search is for
            limit: Maximum number of articles to collect
            
        Returns:
            List of article data dictionaries
        """
        articles = []
        
        try:
            search_results = wikipedia.search(query, results=limit * 2)
            
            for title in search_results[:limit]:
                article = self.collect_article(title, region)
                if article:
                    articles.append(article)
                
                time.sleep(0.5)  # Rate limiting
            
            logger.info(f"Collected {len(articles)} Wikipedia articles for query '{query}'")
            
        except Exception as e:
            logger.error(f"Error searching Wikipedia for '{query}': {e}")
        
        return articles
    
    def collect_region_articles(self, region: str, limit: int = 50) -> list:
        if region not in WIKIPEDIA_CATEGORIES:
            logger.warning(f"No Wikipedia categories configured for region {region}")
            return []
        all_articles = []
        categories = WIKIPEDIA_CATEGORIES[region]
        articles_per_category = max(1, limit // len(categories))
        for category in categories:
            article_titles = self.get_articles_from_category(category, articles_per_category)
            cat_articles = []
            for title in article_titles:
                article = self.collect_article(title, region)
                if article:
                    cat_articles.append(article)
                time.sleep(0.5)
            logger.info(f"Collected {len(cat_articles)} Wikipedia articles from category '{category}'")
            all_articles.extend(cat_articles)
        logger.info(f"Collected {len(all_articles)} Wikipedia articles for region '{region}'")
        return all_articles


class DataCollectionManager:
    """
    Main manager for collecting data from all sources.
    """
    
    def __init__(self):
        self.reddit_collector = RedditCollector()
        self.news_collector = NewsCollector()
        self.wikipedia_collector = WikipediaCollector()
    
    def collect_all_data(self, region: str, save_path: str = None) -> Dict:
        """
        Collect data from all sources for a specific region.
        
        Args:
            region: Region name
            save_path: Path to save collected data
            
        Returns:
            Dictionary containing all collected data
        """
        logger.info(f"Starting data collection for region: {region}")
        
        # Collect from different sources
        reddit_data = self.reddit_collector.collect_region_data(
            region, DATA_LIMITS['reddit_posts']
        )
        
        news_data = self.news_collector.collect_region_news(
            region, DATA_LIMITS['news_articles']
        )
        
        wikipedia_data = self.wikipedia_collector.collect_region_articles(
            region, DATA_LIMITS['wikipedia_articles']
        )
        
        # Combine all data
        all_data = {
            'region': region,
            'reddit': reddit_data,
            'news': news_data,
            'wikipedia': wikipedia_data,
            'collection_metadata': {
                'total_reddit_posts': len(reddit_data['posts']),
                'total_reddit_comments': len(reddit_data['comments']),
                'total_news_articles': len(news_data),
                'total_wikipedia_articles': len(wikipedia_data),
                'collected_at': datetime.now().isoformat()
            }
        }
        
        # Save data if path provided
        if save_path:
            self.save_data(all_data, save_path)
        
        logger.info(f"Data collection completed for region: {region}")
        return all_data
    
    def save_data(self, data: Dict, filepath: str):
        """
        Save collected data to a JSON file.
        
        Args:
            data: Data to save
            filepath: Path to save file
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")
    
    def load_data(self, filepath: str) -> Optional[Dict]:
        """
        Load previously collected data from a JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded data dictionary or None if failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Data loaded from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return None
    
    def collect_all_regions(self, save_dir: str = None) -> Dict:
        """
        Collect data for all configured regions.
        
        Args:
            save_dir: Directory to save collected data
            
        Returns:
            Dictionary containing data for all regions
        """
        all_regions_data = {}
        
        for region in REDDIT_SUBREDDITS.keys():
            logger.info(f"Collecting data for region: {region}")
            
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"{region}_data.json")
            
            region_data = self.collect_all_data(region, save_path)
            all_regions_data[region] = region_data
            
            # Wait between regions to avoid rate limiting
            time.sleep(2)
        
        return all_regions_data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the data collection manager
    collector = DataCollectionManager()
    
    # Collect data for a specific region
    region = 'us_south'
    data = collector.collect_all_data(region)
    
    print(f"Collected data for {region}:")
    print(f"Reddit posts: {len(data['reddit']['posts'])}")
    print(f"Reddit comments: {len(data['reddit']['comments'])}")
    print(f"News articles: {len(data['news'])}")
    print(f"Wikipedia articles: {len(data['wikipedia'])}")
