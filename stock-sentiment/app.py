from src.reddit_sentiment import RedditSentiment
from src.alpaca_api import AlpacaAPI
import pandas as pd

alpaca_api = AlpacaAPI()
top_stocks = alpaca_api.get_top_stocks()

print(top_stocks)

sentiments = []

for stock in top_stocks:
    sentiment = RedditSentiment(stock).get_sentiment_for_symbol()
    if sentiment is None:
        continue
    sentiments.append({
        'symbol': stock,
        'average_sentiment': sentiment['average_sentiment'],
        'sentiment_counts': sentiment['sentiment_counts'],
        'top_posts': sentiment['top_posts'],
        'posts_count': sentiment['posts_count']
    })

df = pd.DataFrame(sentiments)
print(df.head())
