import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import json
import os
from datetime import datetime


from config.data_config import (
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, LANGUAGE_FILTER,
    REMOVE_URLS, REMOVE_MENTIONS, REMOVE_HASHTAGS,
    CLEAN_WHITESPACE, GEOGRAPHIC_MARKERS
)

# Define REGIONS if not imported
REGIONS = {
    'us_south': 'US_SOUTH',
    'uk': 'UK',
    'australia': 'AUSTRALIA',
    'india': 'INDIA',
    'nigeria': 'NIGERIA'
}

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Text preprocessing utilities for GeoLingua project.
    Handles cleaning, filtering, and preparing text data for training.
    """

    def __init__(self):
        self.setup_nltk()
        self.stopwords = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)

    def setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        if CLEAN_WHITESPACE:
            text = re.sub(r'\s+', ' ', text).strip()

        # Remove URLs
        if REMOVE_URLS:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove mentions
        if REMOVE_MENTIONS:
            text = re.sub(r'@[A-Za-z0-9_]+', '', text)

        # Remove hashtags
        if REMOVE_HASHTAGS:
            text = re.sub(r'#[A-Za-z0-9_]+', '', text)

        # Remove Reddit-specific formatting
        text = re.sub(r'/u/[A-Za-z0-9_]+', '', text)  # Remove user mentions
        text = re.sub(r'/r/[A-Za-z0-9_]+', '', text)  # Remove subreddit mentions
        text = re.sub(r'\[deleted\]', '', text)
        text = re.sub(r'\[removed\]', '', text)

        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)

        # Clean up whitespace again
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def filter_text_length(self, text: str) -> bool:
        """
        Filter text based on length criteria.

        Args:
            text: Text to filter

        Returns:
            True if text passes length filter
        """
        if not text:
            return False

        text_len = len(text.strip())
        return MIN_TEXT_LENGTH <= text_len <= MAX_TEXT_LENGTH

    def detect_language(self, text: str) -> str:
        """
        Simple language detection (placeholder for now).

        Args:
            text: Text to analyze

        Returns:
            Language code
        """
        # Simple heuristic - assume English if contains common English words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        words = set(text.lower().split())

        if len(words.intersection(english_words)) > 0:
            return 'en'
        else:
            return 'unknown'

    def extract_geographic_markers(self, text: str) -> Dict[str, List[str]]:
        """
        Extract geographic markers from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of found markers by region
        """
        found_markers = {}
        text_lower = text.lower()

        for region, markers in GEOGRAPHIC_MARKERS.items():
            region_markers = []
            for marker in markers:
                if marker.lower() in text_lower:
                    region_markers.append(marker)

            if region_markers:
                found_markers[region] = region_markers

        return found_markers

    def score_geographic_relevance(self, text: str) -> Dict[str, float]:
        """
        Score text relevance to different geographic regions.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of relevance scores by region
        """
        scores = {}
        markers = self.extract_geographic_markers(text)

        for region in REGIONS.keys():
            if region in markers:
                # Score based on number of markers and their frequency
                score = len(markers[region]) / len(text.split()) * 100
                scores[region] = min(score, 1.0)  # Cap at 1.0
            else:
                scores[region] = 0.0

        return scores

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text.lower())
            # Remove punctuation and stopwords
            tokens = [token for token in tokens if token not in self.punctuation and token not in self.stopwords]
            return tokens
        except Exception as e:
            logger.warning(f"Error tokenizing text: {e}")
            return text.lower().split()

    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        try:
            sentences = sent_tokenize(text)
            return [sent.strip() for sent in sentences if sent.strip()]
        except Exception as e:
            logger.warning(f"Error tokenizing sentences: {e}")
            return [text]


class RedditPreprocessor:
    """
    Specialized preprocessor for Reddit data.
    """

    def __init__(self):
        self.text_preprocessor = TextPreprocessor()

    def preprocess_reddit_post(self, post_data: Dict) -> Optional[Dict]:
        """
        Preprocess a Reddit post.

        Args:
            post_data: Raw Reddit post data

        Returns:
            Processed post data or None if filtered out
        """
        try:
            # Extract relevant fields
            title = post_data.get('title', '')
            text = post_data.get('selftext', '')
            score = post_data.get('score', 0)
            num_comments = post_data.get('num_comments', 0)
            subreddit = post_data.get('subreddit', '')
            created_utc = post_data.get('created_utc', 0)

            # Combine title and text
            combined_text = f"{title}\n{text}".strip()

            # Clean text
            cleaned_text = self.text_preprocessor.clean_text(combined_text)

            # Filter by length
            if not self.text_preprocessor.filter_text_length(cleaned_text):
                return None

            # Filter by language
            language = self.text_preprocessor.detect_language(cleaned_text)
            if language not in LANGUAGE_FILTER:
                return None

            # Extract geographic markers
            geo_markers = self.text_preprocessor.extract_geographic_markers(cleaned_text)
            geo_scores = self.text_preprocessor.score_geographic_relevance(cleaned_text)

            return {
                'text': cleaned_text,
                'original_title': title,
                'original_text': text,
                'score': score,
                'num_comments': num_comments,
                'subreddit': subreddit,
                'created_utc': created_utc,
                'created_datetime': datetime.fromtimestamp(created_utc).isoformat(),
                'language': language,
                'geographic_markers': geo_markers,
                'geographic_scores': geo_scores,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text)
            }

        except Exception as e:
            logger.error(f"Error preprocessing Reddit post: {e}")
            return None

    def preprocess_reddit_comment(self, comment_data: Dict) -> Optional[Dict]:
        """
        Preprocess a Reddit comment.

        Args:
            comment_data: Raw Reddit comment data

        Returns:
            Processed comment data or None if filtered out
        """
        try:
            # Extract relevant fields
            text = comment_data.get('body', '')
            score = comment_data.get('score', 0)
            subreddit = comment_data.get('subreddit', '')
            created_utc = comment_data.get('created_utc', 0)

            # Skip deleted/removed comments
            if text in ['[deleted]', '[removed]', '']:
                return None

            # Clean text
            cleaned_text = self.text_preprocessor.clean_text(text)

            # Filter by length
            if not self.text_preprocessor.filter_text_length(cleaned_text):
                return None

            # Filter by language
            language = self.text_preprocessor.detect_language(cleaned_text)
            if language not in LANGUAGE_FILTER:
                return None

            # Extract geographic markers
            geo_markers = self.text_preprocessor.extract_geographic_markers(cleaned_text)
            geo_scores = self.text_preprocessor.score_geographic_relevance(cleaned_text)

            return {
                'text': cleaned_text,
                'original_text': text,
                'score': score,
                'subreddit': subreddit,
                'created_utc': created_utc,
                'created_datetime': datetime.fromtimestamp(created_utc).isoformat(),
                'language': language,
                'geographic_markers': geo_markers,
                'geographic_scores': geo_scores,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text)
            }

        except Exception as e:
            logger.error(f"Error preprocessing Reddit comment: {e}")
            return None


class NewsPreprocessor:
    """
    Specialized preprocessor for news articles.
    """

    def __init__(self):
        self.text_preprocessor = TextPreprocessor()

    def preprocess_news_article(self, article_data: Dict) -> Optional[Dict]:
        """
        Preprocess a news article.

        Args:
            article_data: Raw news article data

        Returns:
            Processed article data or None if filtered out
        """
        try:
            # Extract relevant fields
            title = article_data.get('title', '')
            text = article_data.get('text', '')
            url = article_data.get('url', '')
            publish_date = article_data.get('publish_date', '')
            authors = article_data.get('authors', [])

            # Combine title and text
            combined_text = f"{title}\n{text}".strip()

            # Clean text
            cleaned_text = self.text_preprocessor.clean_text(combined_text)

            # Filter by length
            if not self.text_preprocessor.filter_text_length(cleaned_text):
                return None

            # Filter by language
            language = self.text_preprocessor.detect_language(cleaned_text)
            if language not in LANGUAGE_FILTER:
                return None

            # Extract geographic markers
            geo_markers = self.text_preprocessor.extract_geographic_markers(cleaned_text)
            geo_scores = self.text_preprocessor.score_geographic_relevance(cleaned_text)

            return {
                'text': cleaned_text,
                'original_title': title,
                'original_text': text,
                'url': url,
                'publish_date': publish_date,
                'authors': authors,
                'language': language,
                'geographic_markers': geo_markers,
                'geographic_scores': geo_scores,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text)
            }

        except Exception as e:
            logger.error(f"Error preprocessing news article: {e}")
            return None


class DatasetPreprocessor:
    """
    Main preprocessor for creating training datasets.
    """

    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.reddit_preprocessor = RedditPreprocessor()
        self.news_preprocessor = NewsPreprocessor()

    def preprocess_dataset(self, raw_data: List[Dict], data_type: str) -> List[Dict]:
        """
        Preprocess a dataset based on its type.

        Args:
            raw_data: List of raw data items
            data_type: Type of data ('reddit_posts', 'reddit_comments', 'news_articles')

        Returns:
            List of preprocessed data items
        """
        processed_data = []

        for item in raw_data:
            try:
                if data_type == 'reddit_posts':
                    processed_item = self.reddit_preprocessor.preprocess_reddit_post(item)
                elif data_type == 'reddit_comments':
                    processed_item = self.reddit_preprocessor.preprocess_reddit_comment(item)
                elif data_type == 'news_articles':
                    processed_item = self.news_preprocessor.preprocess_news_article(item)
                else:
                    logger.warning(f"Unknown data type: {data_type}")
                    continue

                if processed_item:
                    processed_data.append(processed_item)

            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue

        logger.info(f"Preprocessed {len(processed_data)} items from {len(raw_data)} raw items")
        return processed_data

    def create_training_examples(self, processed_data: List[Dict], region: str) -> List[Dict]:
        """
        Create training examples from processed data.
        
        Args:
            processed_data: List of preprocessed data items
            region: Geographic region
            
        Returns:
            List of training examples
        """
        training_examples = []
        
        for item in processed_data:
            try:
                # Handle different data structures
                if 'text' in item:
                    text = item['text']
                elif 'content' in item:
                    # Wikipedia articles have 'content' field
                    text = item['content']
                else:
                    logger.warning(f"Item missing text field: {list(item.keys())}")
                    continue
                
                # Add region markers to create geographic context
                input_text = f"[{REGIONS[region].upper()}] {text}"
                
                # For language model training, output is the same as input
                output_text = text
                
                # Determine source type
                source_type = item.get('subreddit', item.get('url', item.get('title', 'unknown')))
                if not source_type or source_type == '':
                    source_type = 'unknown'
                
                # Create training example
                example = {
                    'input': input_text,
                    'output': output_text,
                    'region': region,
                    'source_type': source_type,
                    'geographic_scores': item.get('geographic_scores', {}),
                    'word_count': item.get('word_count', 0),
                    'metadata': {
                        'score': item.get('score', 0),
                        'created_datetime': item.get('created_datetime', ''),
                        'geographic_markers': item.get('geographic_markers', {})
                    }
                }
                
                training_examples.append(example)
                
            except Exception as e:
                logger.error(f"Error creating training example: {e}")
                continue
        
        return training_examples

    def balance_dataset(self, training_examples: List[Dict], max_per_region: int = 1000) -> List[Dict]:
        """
        Balance dataset across regions.

        Args:
            training_examples: List of training examples
            max_per_region: Maximum examples per region

        Returns:
            Balanced dataset
        """
        # Group by region
        region_examples = {}
        for example in training_examples:
            region = example['region']
            if region not in region_examples:
                region_examples[region] = []
            region_examples[region].append(example)

        # Balance by taking up to max_per_region from each region
        balanced_examples = []
        for region, examples in region_examples.items():
            if len(examples) > max_per_region:
                # Sort by geographic score and take top examples
                examples.sort(key=lambda x: max(x['geographic_scores'].values()) if x['geographic_scores'] else 0, reverse=True)
                examples = examples[:max_per_region]

            balanced_examples.extend(examples)
            logger.info(f"Region {region}: {len(examples)} examples")

        return balanced_examples

    def save_processed_dataset(self, processed_data: List[Dict], output_path: str) -> None:
        """
        Save processed dataset to file.

        Args:
            processed_data: Processed dataset
            output_path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(processed_data)} examples to {output_path}")

        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

    def load_processed_dataset(self, input_path: str) -> List[Dict]:
        """
        Load processed dataset from file.

        Args:
            input_path: Input file path

        Returns:
            Processed dataset
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

            logger.info(f"Loaded {len(processed_data)} examples from {input_path}")
            return processed_data

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise


def main():
    """Test the preprocessors."""
    # Test text preprocessor
    text_preprocessor = TextPreprocessor()

    sample_text = "Hey y'all! Just moved to Atlanta and loving the sweet tea here. Anyone know good BBQ spots? #Atlanta @foodie"

    print("Original text:", sample_text)
    print("Cleaned text:", text_preprocessor.clean_text(sample_text))
    print("Geographic markers:", text_preprocessor.extract_geographic_markers(sample_text))
    print("Geographic scores:", text_preprocessor.score_geographic_relevance(sample_text))

    # Test Reddit preprocessor
    reddit_preprocessor = RedditPreprocessor()

    sample_post = {
        'title': 'Best BBQ in Atlanta?',
        'selftext': 'Y\'all, I need recommendations for the best barbecue in Atlanta. Just moved here from up north.',
        'score': 25,
        'num_comments': 15,
        'subreddit': 'Atlanta',
        'created_utc': 1640995200
    }

    processed_post = reddit_preprocessor.preprocess_reddit_post(sample_post)
    print("\nProcessed Reddit post:")
    print(json.dumps(processed_post, indent=2))


if __name__ == "__main__":
    main()