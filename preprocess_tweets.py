import pandas as pd
import re

tweets = pd.read_csv('TS_Harvey_Tweets.csv')  # or JSON
print(tweets.head())

# print(tweets.columns)
def clean_text(text):
    # Remove URLs and pic.twitter.com links
    text = re.sub(r'https?://\S+|pic\.twitter\.com/\S+', '', text) # remove URLs
    # text = re.sub(r'http\S+', '', text)        
    text = re.sub(r'@\w+', '', text)           # remove @mentions
    text = re.sub(r'#', '', text)              # remove hashtags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text) # remove non-alphanumeric
    return text.lower().strip()

tweets['clean_tweet'] = tweets['Tweet'].astype(str).apply(clean_text)

# Remove duplicate tweets based on cleaned text
tweets = tweets.drop_duplicates(subset='clean_tweet')


tweets = tweets[tweets['clean_tweet'].str.strip().astype(bool)]


tweets['Time'] = pd.to_datetime(tweets['Time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
tweets = tweets[tweets['Time'] >= '2017-01-01']

tweets = tweets[tweets['clean_tweet'].str.split().str.len() >= 5]

tweets.to_csv('Hurricane_Harvey_clean.csv', index=False)
# Print the number of remaining tweets
print(f"Number of valid tweets: {len(tweets)}")