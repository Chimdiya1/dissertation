import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

# Load data
df = pd.read_csv("Hurricane_Harvey_clean.csv")

# Ensure clean text exists
df['clean_tweet'] = df['clean_tweet'].astype(str)

# -----------------------------
# Sentiment Analysis (VADER)
# -----------------------------
sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    scores = sia.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']

df[['neg', 'neu', 'pos', 'compound']] = df['clean_tweet'].apply(
    lambda x: pd.Series(get_sentiment_scores(x))
)

# -----------------------------
# Emotion Analysis (NRCLex)
# -----------------------------
def extract_primary_emotion(text):
    emo = NRCLex(text)
    if emo.top_emotions:
        return emo.top_emotions[0][0]  # highest scoring emotion
    else:
        return "none"

df['primary_emotion'] = df['clean_tweet'].apply(extract_primary_emotion)

# -----------------------------
# Time-based Sentiment Trend
# -----------------------------
df['Time'] = pd.to_datetime(df['Time'])
df['Date'] = df['Time'].dt.date

daily_sentiment = df.groupby('Date')['compound'].mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_sentiment.index, daily_sentiment.values)
plt.title("Average Daily Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Compound Sentiment Score")
plt.tight_layout()
plt.show()

# Save processed data
df.to_csv("tweets_with_sentiment_emotion.csv", index=False)

print("Analysis complete. Saved as tweets_with_sentiment_emotion.csv")
