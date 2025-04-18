import praw
import pandas as pd
from textblob import TextBlob
import re
import os

class RedditSentiment:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'), client_secret=os.getenv('REDDIT_SECRET_KEY'), user_agent=os.getenv('REDDIT_USER_AGENT'))

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def get_posts(self, limit: int = 100, time_filter: str = 'month'):
        try:
            posts = self.reddit.subreddit('wallstreetbets+stock+investing').search(f'{self.symbol} stock', limit=limit, time_filter=time_filter)
            posts_list = list(posts)
            if len(posts_list) == 0:
                raise Exception(f'No posts found for {self.symbol}')
            return posts_list
        except Exception as e:
            raise e

    def get_sentiment(self, text: str) -> float:
        text = self.clean_text(text)
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def clean_text(self, text: str) -> str:
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove Reddit-style links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def get_sentiment(self, text: str) -> float:
        if text is None:
            return 0
        text = self.clean_text(text)
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def create_df(self, posts: list[praw.models.Submission]) -> pd.DataFrame:
        data = []
        for post in posts:
            data.append({
                'postId': post.id,
                'title': post.title,
                'text': post.selftext,
                'sentiment': self.get_sentiment(post.title + ' ' + post.selftext),
                'created': post.created,
                'url': post.url,
                'author': post.author.name,
            })
        
        df = pd.DataFrame(data)
        return df

    def analyize_sentiment(self, df: pd.DataFrame) -> dict:
        avg_sent = df['sentiment'].mean()
        df['sentiment_category'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
        sentiment_counts = df['sentiment_category'].value_counts().to_dict()
        top_posts = df.nlargest(5, 'sentiment').to_dict(orient='records')
        
        return {
            'average_sentiment': float(avg_sent),
            'sentiment_counts': sentiment_counts,
            'top_posts': top_posts,
            'posts_count': len(df)
        }

    def get_sentiment_for_symbol(self) -> dict:
        try:
            posts = self.get_posts(limit=100, time_filter='month')
            df = self.create_df(posts)
            return self.analyize_sentiment(df)
        except Exception as e:
            print(e)
            return None
