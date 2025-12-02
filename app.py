import streamlit as st
from recommender import UserRecommender

# ---------------------------
# Initialize Session State
# ---------------------------
if "rec" not in st.session_state:
    st.session_state.rec = UserRecommender()
if "songA" not in st.session_state:
    st.session_state.songA = None
    st.session_state.songB = None
if "final_song" not in st.session_state:
    st.session_state.final_song = None


# ---------------------------
# Helper: Show a Song Card
# ---------------------------
def show_song_card(song_info, key):
    st.image(song_info["thumbnail"], width=200)
    st.write(f"### {song_info['title']}")
    st.write(f"**{song_info['artist']}**")
    return st.button(f"Select", key=key)


# ---------------------------
# UI HEADER
# ---------------------------
st.title("🎵 This-or-That Music Recommender")
st.write("Pick between two songs → I learn your taste → I recommend one!")

# ---------------------------
# If already recommended → show final result
# ---------------------------
if st.session_state.final_song:
    st.success("🎉 Your recommendation is ready!")
    song = st.session_state.rec.get_song_info(st.session_state.final_song)

    st.image(song["thumbnail"], width=300)
    st.markdown(f"## **{song['title']}**")
    st.markdown(f"### *{song['artist']}*")

    if st.button("Restart"):
        st.session_state.rec = UserRecommender()
        st.session_state.songA = None
        st.session_state.songB = None
        st.session_state.final_song = None
    st.stop()

# ---------------------------
# Pick two new songs if none loaded
# ---------------------------
if st.session_state.songA is None:
    s1, s2 = st.session_state.rec.pick_two_songs()
    st.session_state.songA = s1
    st.session_state.songB = s2

songA = st.session_state.rec.get_song_info(st.session_state.songA)
songB = st.session_state.rec.get_song_info(st.session_state.songB)

# ---------------------------
# Show two songs side-by-side
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    if show_song_card(songA, "pickA"):
        st.session_state.rec.choose(songA["id"])
        st.session_state.songA = None
        st.session_state.songB = None

with col2:
    if show_song_card(songB, "pickB"):
        st.session_state.rec.choose(songB["id"])
        st.session_state.songA = None
        st.session_state.songB = None

# ---------------------------
# After 10–15 choices → Recommend
# ---------------------------
st.write("---")
st.write(f"### ⭐ Choices so far: {st.session_state.rec.steps}")

if st.session_state.rec.steps >= 10:
    if st.button("🎯 Get Final Recommendation"):
        final_id = st.session_state.rec.recommend(top_k=1)[0]
        st.session_state.final_song = final_id
        st.rerun()