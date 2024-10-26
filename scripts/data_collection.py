import tweepy
import pandas as pd
from config.credentials import api_key, api_secret, access_token, access_token_secret

# Authenticate to Twitter API
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def collect_tweets(topic, count=100):
    tweets = api.search_tweets(q=topic, count=count, lang="en")
    data = [{"text": tweet.text, "user": tweet.user.screen_name, "location": tweet.user.location} for tweet in tweets]
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = collect_tweets("example topic", 100)
    df.to_csv('data/raw_data/tweets.csv', index=False)
