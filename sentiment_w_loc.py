import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------------
# Load Data
# ---------------------------
tweets = pd.read_csv("tweets_with_sentiment_emotion.csv")

# ---------------------------
# Load spaCy NER
# ---------------------------
nlp = spacy.load("en_core_web_lg")

# ---------------------------
# 1. Manual hurricane location keywords
# ---------------------------
location_keywords = {
    "texas", "houston", "galveston", "corpus christi", "rockport", "louisiana",
    "florida", "gulf coast", "caribbean", "dominica", "grenada", "barbados",
    "puerto rico", "new orleans", "mississippi"
}

# Normalize keyword list for comparison
location_keywords = {loc.lower() for loc in location_keywords}


# ---------------------------
# 2. Function: extract spaCy NER locations
# ---------------------------
def extract_ner_locations(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents if ent.label_ in ("GPE", "LOC")]


# ---------------------------
# 3. Function: extract keyword locations
# ---------------------------
def extract_keyword_locations(text):
    text_lower = text.lower()
    found = []

    for loc in location_keywords:
        if loc in text_lower:
            found.append(loc)

    return found


# ---------------------------
# 4. Combine methods
# ---------------------------
def extract_all_locations(text):
    ner_locs = extract_ner_locations(text)
    keyword_locs = extract_keyword_locations(text)

    all_locs = set(ner_locs + keyword_locs)  # remove duplicates
    return list(all_locs)


# ---------------------------
# Apply extraction to tweets
# ---------------------------
tweets["locations"] = tweets["clean_tweet"].astype(str).apply(extract_all_locations)

# Keep only tweets mentioning some location
tweets_with_locations = tweets[tweets["locations"].str.len() > 0]

tweets.to_csv("tweets_with_loc.csv")
print(tweets_with_locations[["clean_tweet", "locations"]].head())



# ---------------------------
# 5. Count location frequencies
# ---------------------------
from collections import Counter

location_counter = Counter()

for loc_list in tweets_with_locations["locations"]:
    location_counter.update(loc_list)

top_locations = pd.DataFrame(location_counter.most_common(30),
                             columns=["location", "count"])

print("\nTop detected locations:\n")
print(top_locations)


# ---------------------------
# 6. Visualize top locations
# ---------------------------
plt.figure(figsize=(12, 6))
plt.bar(top_locations["location"], top_locations["count"])
plt.xticks(rotation=45, ha="right")
plt.title("Most Mentioned Locations in Tweets")
plt.xlabel("Location")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()


# # -----------------------------------------
# # 7. Compute average sentiment for top locations
# # -----------------------------------------

# # Ensure compound sentiment exists
# if "compound" not in tweets.columns:
#     raise ValueError("Sentiment scores not found. Run sentiment analysis first.")

top_locs_list = top_locations["location"].tolist()

# location_sentiment = []

# for loc in top_locs_list:
#     subset = tweets_with_locations[tweets_with_locations["locations"].apply(lambda x: loc in x)]
    
#     if len(subset) > 0:
#         avg_sentiment = subset["compound"].mean()
#         tweet_count = len(subset)
#         location_sentiment.append([loc, tweet_count, avg_sentiment])

# sentiment_df = pd.DataFrame(location_sentiment,
#                              columns=["location", "tweet_count", "avg_compound_sentiment"])

# print("\nAverage Sentiment for Top Locations:\n")
# print(sentiment_df)

# # -----------------------------------------
# # 8. Plot sentiment by location (optional)
# # -----------------------------------------
# plt.figure(figsize=(12, 6))
# plt.bar(sentiment_df["location"], sentiment_df["avg_compound_sentiment"])
# plt.xticks(rotation=45, ha="right")
# plt.title("Average Sentiment (Compound) for Top Mentioned Locations")
# plt.xlabel("Location")
# plt.ylabel("Average Compound Score")
# plt.tight_layout()
# plt.show()

# # Save results
# sentiment_df.to_csv("location_sentiment_scores.csv", index=False)


# -----------------------------------------
# 9. Emotion Breakdown (NRCLex) for Top Locations
# -----------------------------------------

emotion_breakdown = []

for loc in top_locs_list:
    subset = tweets_with_locations[tweets_with_locations["locations"].apply(lambda x: loc in x)]

    if len(subset) > 0:
        emotion_counts = subset["primary_emotion"].value_counts()
        total = emotion_counts.sum()

        # Get dominant emotion
        dominant_emotion = emotion_counts.idxmax()
        dominant_percentage = (emotion_counts.max() / total)

        # Store row summary
        emotion_breakdown.append([
            loc,
            total,
            dominant_emotion,
            round(dominant_percentage, 3),
            emotion_counts.to_dict()  # full distribution
        ])

emotion_df = pd.DataFrame(
    emotion_breakdown,
    columns=["location", "tweet_count", "dominant_emotion", "dominant_emotion_pct", "emotion_distribution"]
)

print("\nEmotion Breakdown for Top Locations:\n")
print(emotion_df)

# Save emotion summary
emotion_df.to_csv("location_emotion_breakdown.csv", index=False)

# -----------------------------------------
# 11. Heatmap: Emotions vs. Locations
# -----------------------------------------



# Convert emotion distributions into a matrix
heatmap_data = {}

for _, row in emotion_df.iterrows():
    loc = row["location"]
    dist = row["emotion_distribution"]  # dict: {"fear": 10, "anger": 5, ...}
    heatmap_data[loc] = dist

# Create DataFrame (missing emotions will autofill with NaN → fill 0)
emotion_matrix = pd.DataFrame(heatmap_data).fillna(0).astype(int)

plt.figure(figsize=(14, 8))
sns.heatmap(emotion_matrix, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Emotion Frequency Heatmap per Location")
plt.xlabel("Location")
plt.ylabel("Emotion")
plt.tight_layout()
plt.show()
