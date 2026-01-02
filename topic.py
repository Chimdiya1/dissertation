# # ----------------------------------------
# # BERTopic MODELING FOR TWEETS
# # ----------------------------------------
# import pandas as pd
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer

# # Load data
# df = pd.read_csv("tweets_with_sentiment_emotion.csv")
# texts = df["clean_tweet"].astype(str).tolist()

# # 1. Embeddings model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # 2. Fit BERTopic
# topic_model = BERTopic(
#         embedding_model=model,
#         umap_model=None,        # disable umap
#         hdbscan_model=None,     # use kmeans fallback
#         nr_topics="auto"
# )
# topics, probs = topic_model.fit_transform(texts)

# # 3. Save topics
# df["topic"] = topics
# topic_model.save("bertopic_harvey")

# df.to_csv("tweets_with_topics_bertopic.csv", index=False)

# # 4. Display top topics
# print(topic_model.get_topic_info())

# # 5. Visualize (optional)
# topic_model.visualize_topics().show()
# topic_model.visualize_barchart().show()


import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ---------------------------
# Load data
# ---------------------------
# df = pd.read_csv("tweets_with_sentiment_emotion.csv")
# texts = df['clean_tweet'].astype(str).tolist()

# # ---------------------------
# # Load embedding model
# # ---------------------------
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ---------------------------
# # Generate embeddings sequentially
# # ---------------------------
# embeddings = []
# for text in texts:
#     emb = embedding_model.encode(text, show_progress_bar=False)
#     embeddings.append(emb)

# # Convert to NumPy array
# embeddings = np.array(embeddings)

# # ---------------------------
# # Initialize BERTopic (macOS-safe)
# # ---------------------------
# topic_model = BERTopic(umap_model=None, hdbscan_model=None, nr_topics="auto")

# # ---------------------------
# # Fit-transform using precomputed embeddings
# # ---------------------------
# topics, probs = topic_model.fit_transform(texts, embeddings)

# # ---------------------------
# # Save results
# # ---------------------------
# df['topic'] = topics
# df.to_csv("tweets_with_topics.csv", index=False)
# print("BERTopic analysis complete.")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("tweets_with_sentiment_emotion.csv")
texts = df['clean_tweet'].astype(str).tolist()

# ---------------------------
# Generate embeddings
# ---------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array([embedding_model.encode(t, show_progress_bar=False) for t in texts])

# ---------------------------
# Cluster embeddings
# ---------------------------
num_topics = 10  # adjust as needed
kmeans = KMeans(n_clusters=num_topics, random_state=42)
topics = kmeans.fit_predict(embeddings)
df['topic'] = topics

# ---------------------------
# Extract top words per topic
# ---------------------------
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)
feature_names = np.array(vectorizer.get_feature_names_out())

top_n = 10  # top words per topic
top_words = {}

for topic_id in range(num_topics):
    topic_texts = df[df['topic'] == topic_id]['clean_tweet']
    X_topic = vectorizer.transform(topic_texts)
    word_counts = np.array(X_topic.sum(axis=0)).flatten()
    top_words_indices = word_counts.argsort()[::-1][:top_n]
    top_words[topic_id] = feature_names[top_words_indices]

# ---------------------------
# Display top words
# ---------------------------
print("Top words per topic:\n")
for topic_id, words in top_words.items():
    print(f"Topic {topic_id}: {', '.join(words)}")

# ---------------------------
# Plot number of tweets per topic
# ---------------------------
topic_counts = df['topic'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(topic_counts.index, topic_counts.values, color='skyblue')
plt.xticks(topic_counts.index)
plt.xlabel("Topic ID")
plt.ylabel("Number of Tweets")
plt.title("Tweet Counts per Topic")
plt.tight_layout()
plt.show()

