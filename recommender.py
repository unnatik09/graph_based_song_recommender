import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class UserRecommender:
    def __init__(self, graph_path="song_graph_1.pkl", csv_path="final_songs_clean.csv"):
        # Load graph
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)

        # Load dataframe
        self.df = pd.read_csv(csv_path)

        # Embedding columns
        self.emb_cols = [c for c in self.df.columns if c.startswith("pca_emb_")]

        # Fast lookup
        self.id_to_idx = {row["id"]: i for i, row in self.df.iterrows()}

        # User profile vector (64-dim PCA)
        self.user_vec = np.zeros(len(self.emb_cols))

        # Keep track of seen songs
        self.seen = set()

        # How many choices user has made
        self.steps = 0


    # -----------------------------------------------------------
    # Convert song ID → 64-dim embedding vector
    # -----------------------------------------------------------
    def get_vector(self, song_id):
        idx = self.id_to_idx[song_id]
        return self.df.loc[idx, self.emb_cols].values.astype(float)


    # -----------------------------------------------------------
    # Pick 2 songs to show user (This vs That)
    # -----------------------------------------------------------
    def pick_two_songs(self):
        """
        Warmup (first 6 picks = 12 songs):
            - choose 12 songs
            - each from a different language
            - and different category
            - and embeddings different (low similarity)
        Adaptive:
            - similarity to user_vec
        """

        all_ids = list(self.G.nodes())
        unseen_ids = [s for s in all_ids if s not in self.seen]

        # -------------------------------------------------------
        # WARMUP PHASE — FIRST 6 ROUNDS = 12 SONGS
        # -------------------------------------------------------
        if self.steps < 6:

            # Build mapping: language → songs
            lang_map = {}
            for i, row in self.df.iterrows():
                lang = row["language_encoded"]
                if lang not in lang_map:
                    lang_map[lang] = []
                lang_map[lang].append(row["id"])

            # Sort languages so selection is deterministic
            langs = sorted(lang_map.keys())

            warmup_list = []

            # STEP 1: pick 1 song per language (first pass)
            for lang in langs:
                for sid in lang_map[lang]:
                    if sid in unseen_ids:
                        warmup_list.append(sid)
                        break
                if len(warmup_list) >= 12:
                    break

            # If fewer than 12 languages exist:
            # STEP 2: pick additional songs from remaining languages,
            # but avoid similarity
            if len(warmup_list) < 12:
                needed = 12 - len(warmup_list)

                # collect all songs sorted by language blocks
                remaining = [sid for sid in unseen_ids if sid not in warmup_list]

                # we add songs that are FAR from existing warmup songs
                def far_enough(song_id):
                    v_cand = self.get_vector(song_id)
                    for sid in warmup_list:
                        v_old = self.get_vector(sid)
                        sim = cosine_similarity(
                            v_cand.reshape(1, -1),
                            v_old.reshape(1, -1)
                        )[0][0]
                        if sim > 0.45:  # too similar, reject
                            return False
                    return True

                for sid in remaining:
                    if far_enough(sid):
                        warmup_list.append(sid)
                    if len(warmup_list) >= 12:
                        break

            # ---- map warmup picks to rounds (2 songs per round) ----
            idx1 = self.steps * 2
            idx2 = idx1 + 1

            if idx2 < len(warmup_list):
                return warmup_list[idx1], warmup_list[idx2]
            else:
                # fallback random pick
                return np.random.choice(unseen_ids, size=2, replace=False)

        # -------------------------------------------------------
        # ADAPTIVE PHASE — AFTER WARMUP (step >= 6)
        # -------------------------------------------------------
        vectors = np.array([self.get_vector(s) for s in unseen_ids])
        scores = cosine_similarity(vectors, self.user_vec.reshape(1, -1)).flatten()

        # Pick top 30
        top_idx = scores.argsort()[::-1][:30]
        candidates = [unseen_ids[i] for i in top_idx]

        # pick most similar
        c1 = candidates[0]
        v1 = self.get_vector(c1)

        # pick least similar from pool (diversity)
        cand_vecs = np.array([self.get_vector(c) for c in candidates])
        dist = cosine_similarity(v1.reshape(1, -1), cand_vecs).flatten()
        c2 = candidates[np.argmin(dist)]

        return c1, c2

        # -------------------------------------------------------
        # ADAPTIVE PHASE
        # -------------------------------------------------------
        vectors = np.array([self.get_vector(s) for s in unseen_ids])
        scores = cosine_similarity(vectors, self.user_vec.reshape(1, -1)).flatten()

        top_indices = scores.argsort()[::-1][:30]
        candidates = [unseen_ids[i] for i in top_indices]

        # pick the most preferred candidate
        c1 = candidates[0]
        v1 = self.get_vector(c1)

        # pick the *least similar* among the candidate pool
        cand_vecs = np.array([self.get_vector(c) for c in candidates])
        sim_to_c1 = cosine_similarity(v1.reshape(1, -1), cand_vecs).flatten()

        farthest_idx = np.argmin(sim_to_c1)
        c2 = candidates[farthest_idx]

        return c1, c2


    # -----------------------------------------------------------
    # User chooses one of the two songs
    # -----------------------------------------------------------
    def choose(self, chosen_song_id):
        """
        Update user taste profile based on chosen song.
        """
        vec = self.get_vector(chosen_song_id)

        # Weighted update
        self.user_vec = 0.9 * self.user_vec + 0.1 * vec

        # Normalize
        if np.linalg.norm(self.user_vec) > 0:
            self.user_vec /= np.linalg.norm(self.user_vec)

        self.seen.add(chosen_song_id)
        self.steps += 1


    # -----------------------------------------------------------
    # Final Recommendation
    # -----------------------------------------------------------
    def recommend(self, top_k=1):
        """
        Recommend song(s) most aligned with user taste.
        Uses cosine similarity + graph connectivity as tie-breaker.
        """
        all_ids = list(self.G.nodes())

        unseen_ids = [s for s in all_ids if s not in self.seen]
        vectors = np.array([self.get_vector(s) for s in unseen_ids])

        # Similarity to user vector
        scores = cosine_similarity(vectors, self.user_vec.reshape(1, -1)).flatten()

        # Rank by score
        top_indices = scores.argsort()[::-1][:top_k]
        recommendations = [unseen_ids[i] for i in top_indices]

        return recommendations


    # -----------------------------------------------------------
    # Helper: Pretty print a song
    # -----------------------------------------------------------
    def get_song_info(self, song_id):
        row = self.df.iloc[self.id_to_idx[song_id]]
        return {
            "id": row["id"],
            "title": row["title"],
            "artist": row["artist"],
            "album": row["album"],
            "thumbnail": row["thumbnail"]
        }