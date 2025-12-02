import pandas as pd
import networkx as nx
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import ast

# ------------------------------------
# LOAD DATA
# ------------------------------------
df = pd.read_csv("final_songs_clean.csv")

# Fix related_videos from string to list
def fix_list(x):
    try:
        if isinstance(x, list):
            return x
        return ast.literal_eval(x)
    except:
        return []
df["related_videos"] = df["related_videos"].apply(fix_list)

# ------------------------------------
# FEATURES USED FOR SIMILARITY
# ------------------------------------
embedding_cols = [c for c in df.columns if c.startswith("pca_emb_")]

metadata_cols = [
    "popularity",
    "length_seconds",
    "release_year",
    "language_encoded",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "category_encoded",
    "genre_cluster",
    "available_countries"
]

# Numeric matrix
X_emb = df[embedding_cols].values               # PCA embeddings
X_meta = df[metadata_cols].values               # metadata


# ------------------------------------
# COSINE SIMILARITY FOR EMBEDDINGS
# ------------------------------------
print("Computing cosine similarity on embeddings...")
sim_emb = cosine_similarity(X_emb)

# Normalize between 0–1
sim_emb = (sim_emb - sim_emb.min()) / (sim_emb.max() - sim_emb.min())


# ------------------------------------
# METADATA SIMILARITY
# ------------------------------------
print("Computing cosine similarity on metadata...")
sim_meta = cosine_similarity(X_meta)

sim_meta = (sim_meta - sim_meta.min()) / (sim_meta.max() - sim_meta.min())


# ------------------------------------
# HYBRID SIMILARITY
# ------------------------------------
# You can tune these weights
WEIGHT_EMB = 0.6
WEIGHT_META = 0.4

print("Combining similarities...")
sim_hybrid = WEIGHT_EMB * sim_emb + WEIGHT_META * sim_meta


# ------------------------------------
# BUILD GRAPH
# ------------------------------------
print("Building graph with hybrid similarity + related_videos edges...")

G = nx.Graph()

# Add nodes first
for idx, row in df.iterrows():
    song_id = row["id"]

    # skip invalid song ids
    if pd.isna(song_id) or not isinstance(song_id, str) or song_id.strip() == "":
        continue

    G.add_node(song_id,
               title=row["title"],
               artist=row["artist"],
               album=row["album"],
               thumbnail=row["thumbnail"],
               idx=idx)

# ------------------------------------
# ADD SIMILARITY EDGES
# ------------------------------------
N = len(df)
THRESHOLD = 0.3   # only add edges above this similarity

print("Adding similarity edges...")
for i in tqdm(range(N)):
    for j in range(i + 1, N):
        weight = sim_hybrid[i, j]
        if weight >= THRESHOLD:
            G.add_edge(df["id"][i], df["id"][j], weight=float(weight))


# ------------------------------------
# ADD RELATED VIDEO EDGES
# ------------------------------------
print("Adding related_videos edges...")

related_weight = 0.9  # stronger weight

video_to_index = {v: i for i, v in enumerate(df["id"])}

for idx, row in df.iterrows():
    src = row["id"]
    for vid in row["related_videos"]:
        if vid in video_to_index:
            tgt = vid
            # add or update edge with max weight
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] = max(G[src][tgt]["weight"], related_weight)
            else:
                G.add_edge(src, tgt, weight=related_weight)


# ------------------------------------
# SAVE GRAPH
# ------------------------------------
with open("song_graph_1.pkl", "wb") as f:
    pickle.dump(G, f)

print("--------------------------------------------------")
print("Graph built successfully!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Saved as song_graph.gpickle")
print("--------------------------------------------------")