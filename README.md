<img width="2940" height="1688" alt="image" src="https://github.com/user-attachments/assets/e573a757-6e37-44a6-8571-5fc57f7bc9c4" />
<img width="2940" height="1688" alt="image" src="https://github.com/user-attachments/assets/65ce401d-52ee-4df3-ac8f-fbdb67e3ed8f" />

# 🎵 Graph-Based Song Recommender

A minimal, proof-of-concept music recommendation system built using graph and embedding techniques.  
This repository implements a “This-or-That → Personalized Recommendation” pipeline — similar in spirit to how Spotify/YouTube Music might infer your taste, but at a small scale (hundreds of songs).

---

## 🔧 Features & Architecture

- **Data gathering**: Uses YouTube Music metadata (title, artist, album, thumbnail) + optional enriched metadata.  
- **Embeddings + PCA**: Converts song metadata into 64-dim numeric embeddings (via text embeddings + dimension reduction).  
- **Hybrid similarity graph**: Combines embedding similarity, metadata similarity and YouTube-Music “related songs” links to build a weighted song graph.  
- **Adaptive “This-or-That” selection**: On first uses, user picks between pairs of songs. These choices build a personal taste vector.  
- **Personalized recommendation**: After a few selections, the system recommends a song matching your taste but not shown before.  
- **Lightweight and extendable**: Everything is in Python; dependencies are minimal; works on small datasets (hundreds of songs).  

---

