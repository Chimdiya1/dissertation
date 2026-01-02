import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from textstat import textstat
from collections import Counter


nltk.download('stopwords')

# Load dataset
df = pd.read_csv("Hurricane_Harvey_clean.csv")
df["clean_tweet"] = df["clean_tweet"].astype(str)

# -----------------------------
# READABILITY SCORES
# -----------------------------
df["flesch_reading_ease"] = df["clean_tweet"].apply(textstat.flesch_reading_ease)
df["flesch_kincaid_grade"] = df["clean_tweet"].apply(textstat.flesch_kincaid_grade)

print(df[["flesch_reading_ease", "flesch_kincaid_grade"]].describe())

# -----------------------------
# LEXICAL DIVERSITY
# -----------------------------
def lexical_diversity(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

df["lexical_diversity"] = df["clean_tweet"].apply(lexical_diversity)
print(df["lexical_diversity"].describe())

# -----------------------------
# REPETITIVE PATTERNS
# -----------------------------

# Most common words
stop_words = set(stopwords.words("english"))
all_words = [w for tweet in df["clean_tweet"] for w in tweet.split() if w not in stop_words]
word_freq = Counter(all_words).most_common(20)
print("Most common words:", word_freq)

# Most common bigrams
bigrams = [bg for tweet in df["clean_tweet"] for bg in ngrams(tweet.split(), 2)]
bigram_freq = Counter(bigrams).most_common(15)
print("Most common bigrams:", bigram_freq)

# Most common trigrams
trigrams = [tg for tweet in df["clean_tweet"] for tg in ngrams(tweet.split(), 3)]
trigram_freq = Counter(trigrams).most_common(10)
print("Most common trigrams:", trigram_freq)

# -----------------------------
# PLOTS
# -----------------------------

# Readability distribution
plt.figure(figsize=(10, 4))
plt.hist(df["flesch_reading_ease"], bins=30)
plt.title("Flesch Reading Ease Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Lexical Diversity distribution
plt.figure(figsize=(10, 4))
plt.hist(df["lexical_diversity"], bins=30)
plt.title("Lexical Diversity Distribution")
plt.xlabel("Unique / Total Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Word frequency bar chart
words, counts = zip(*word_freq)
plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.title("Most Common Words")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
