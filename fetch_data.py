from ytmusicapi import YTMusic
import pandas as pd
import re
import os

yt = YTMusic()

# -------------------------------
# YOUR PLAYLISTS
# -------------------------------
PLAYLIST_URLS = [
    "https://music.youtube.com/playlist?list=PLrCRhdnhmz6a0pMjhCGFA8vBvuxpjyvEI&si=oIAmd-m1P38q36Z0",
    "https://music.youtube.com/playlist?list=PLO7-VO1D0_6M1xUjj8HxTxskouWx48SNw",
    "https://music.youtube.com/playlist?list=PLPzU9LIZMn3KM2clQIieTUD2MyMMSYA2U",
    "https://music.youtube.com/playlist?list=PL7KYHPKrfi5wJtnGJaIK3RvCOwDlB8C6X",
    "https://music.youtube.com/playlist?list=PLOwsX2pyPBjk-bxz6Vf8J1P8CRSdT3C9z",
]

# -------------------------------
# Extract playlist ID
# -------------------------------
def extract_id(url):
    match = re.search(r"list=([^&]+)", url)
    if match:
        return match.group(1)
    return None


# -------------------------------
# Fetch playlist or album
# -------------------------------
def fetch_items(list_id):
    if list_id.startswith("PL"):
        try:
            pl = yt.get_playlist(list_id)
            print(f"  → {len(pl.get('tracks', []))} tracks (playlist)")
            return pl.get("tracks", [])
        except Exception as e:
            print("Skipping invalid playlist:", list_id, "Error:", e)
            return []

    if list_id.startswith("OLAK5uy_"):
        try:
            al = yt.get_album(list_id)
            print(f"  → {len(al.get('tracks', []))} tracks (album/autogen)")
            return al.get("tracks", [])
        except Exception as e:
            print("Skipping invalid album:", list_id, "Error:", e)
            return []

    return []


# -------------------------------
# Fetch advanced metadata from YTMusic
# -------------------------------
def enrich_metadata(video_id):
    """
    Pulls advanced metadata from yt.get_song()
    Everything is guarded with .get() to avoid crashes.
    """
    try:
        data = yt.get_song(video_id)
    except:
        return {}

    micro = data.get("microformat", {}).get("microformatDataRenderer", {})
    details = data.get("videoDetails", {})

    return {
        "views": int(details.get("viewCount", 0)),
        "is_private": details.get("isPrivate"),
        "is_unlisted": details.get("isUnlisted"),
        "category": micro.get("category"),
        "publish_date": micro.get("publishDate"),
        "upload_date": micro.get("uploadDate"),
        "length_seconds": details.get("lengthSeconds"),
        "is_family_safe": micro.get("isFamilySafe"),
        "available_countries": len(micro.get("availableCountries", [])),
    }


# -------------------------------
# Normalize a track
# -------------------------------
def normalize(track):
    return {
        "id": track.get("videoId"),
        "title": track.get("title"),
        "artist": (track.get("artists") or [{}])[0].get("name"),
        "album": (track.get("album") or {}).get("name") if isinstance(track.get("album"), dict) else None,
        "thumbnail": (track.get("thumbnails") or [{}])[-1].get("url"),
        "duration": track.get("duration"),
        "related_videos": [r.get("videoId") for r in (track.get("related") or []) if r.get("videoId")]
    }


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    songs = []

    for url in PLAYLIST_URLS:
        print("\nFetching:", url)
        pid = extract_id(url)
        if not pid:
            print("Could not extract playlist ID:", url)
            continue

        items = fetch_items(pid)
        for t in items:
            if "videoId" in t:
                base = normalize(t)
                adv = enrich_metadata(base["id"])
                songs.append({**base, **adv})

    df = pd.DataFrame(songs).drop_duplicates("id")
    df.to_csv("data/songs_full.csv", index=False)

    print("\n-------------------------------------")
    print("Saved: data/songs_full.csv")
    print("Total songs:", len(df))
    print("-------------------------------------")