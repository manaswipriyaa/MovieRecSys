# ==============================

# CineMatch FINAL (Netflix + AI + Hybrid)

# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, requests, urllib.parse

st.set_page_config(page_title="CineMatch", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------

# SESSION STATE

# ------------------------------

if "tmdb_cache" not in st.session_state:
    st.session_state.tmdb_cache = {}

for k, v in {"page":"home","movie":None,"watchlist":[],"recs":None,"search_q":""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------

# CONSTANTS

# ------------------------------

TMDB_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlMjc5NDdhNTJhOTc2Nzc1MzBmZGQ5ZTQ3NmU4YjE3YyIsIm5iZiI6MTc0NDAyNTU1Mi40NzIsInN1YiI6IjY3ZjNiN2QwYTU0NzFhNTFlZTk5NjFjMCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.jnnqOIMvbzOrkx1Wy6Kll6OP5R5MIGEHHr9mq9124Pc"
IMG_BASE = "https://image.tmdb.org/t/p"

# ------------------------------

# LOAD DATA

# ------------------------------

@st.cache_data
def load_data():
    rd = pd.read_csv("data/ratings.csv")
    md = pd.read_csv("data/movies.csv")
    md['genres'] = md['genres'].fillna('')
    return rd, md

ratings_df, movies_df = load_data()

# FAST LOOKUPS

title_to_genre = dict(zip(movies_df['title'], movies_df['genres']))
movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
title_to_movie_id = dict(zip(movies_df['title'], movies_df['movieId']))

# ------------------------------

# TMDB

# ------------------------------

@st.cache_data
def tmdb_search(title):
    if title in st.session_state.tmdb_cache:
        return st.session_state.tmdb_cache[title]
try:
    r = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={"api_key": TMDB_KEY, "query": title},
        timeout=5
    )
    res = r.json().get('results', [])
    result = res[0] if res else None
    st.session_state.tmdb_cache[title] = result
    return result
except:
    return None

def poster_url(title):
    r = tmdb_search(title)
    if r and r.get("poster_path"):
        return f"{IMG_BASE}/w300{r['poster_path']}"
    return None

# ------------------------------

# TFIDF CACHE

# ------------------------------

@st.cache_data
def build_tfidf(mdf):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    mat = tfidf.fit_transform(mdf['genres'])
    return tfidf, mat

# ------------------------------

# COLLAB MODEL

# ------------------------------

@st.cache_data
def build_collab_model(ratings_df):
    user_movie = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    U, sigma, Vt = np.linalg.svd(user_movie, full_matrices=False)
    sigma = np.diag(sigma)
    return user_movie, np.dot(np.dot(U, sigma), Vt)

def collab_recs(watchlist, user_movie, reconstructed, n=20):
    if not watchlist:
        return pd.DataFrame()

user_vec = np.zeros(user_movie.shape[1])

for t in watchlist:
    mid = title_to_movie_id.get(t)
    if mid in user_movie.columns:
        idx = list(user_movie.columns).index(mid)
        user_vec[idx] = 5

scores = np.dot(user_vec, reconstructed.T)
ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

res = []
for idx, score in ranked[:n*2]:
    mid = user_movie.index[idx]
    title = movie_id_to_title.get(mid)
    if title and title not in watchlist:
        res.append({
            "title": title,
            "genres": title_to_genre.get(title, ""),
            "score": float(score)
        })

return pd.DataFrame(res[:n])

# ------------------------------

# HYBRID

# ------------------------------

def hybrid_recs(picked, watchlist, mdf, n=10):
    tfidf, mat = build_tfidf(mdf)
    cos = cosine_similarity(
    tfidf.transform(['|'.join(picked)]),
    mat).flatten()

def cov(g):
    return len(set(picked) & set(g.split('|'))) / len(picked)

content = 0.7*cos + 0.3*mdf['genres'].apply(cov).values

user_movie, reconstructed = build_collab_model(ratings_df)
collab_df = collab_recs(watchlist, user_movie, reconstructed, n)

collab_map = dict(zip(collab_df['title'], collab_df['score']))
max_cf = max(collab_map.values()) if collab_map else 1

scores = []
for i, row in mdf.iterrows():
    t = row['title']
    cf = collab_map.get(t, 0) / max_cf
    final = 0.6*content[i] + 0.4*cf
    scores.append(final)

mdf['score'] = scores
top = mdf.sort_values(by='score', ascending=False).head(n)

return top[['title','genres','score']]

# ------------------------------

# AI MOOD

# ------------------------------

MOOD_MAP = {
"sad": ["Drama","Romance"],
"funny": ["Comedy"],
"action": ["Action"],
"romantic": ["Romance"],
"scary": ["Horror","Thriller"],
"space": ["Sci-Fi"],
}

def mood_to_genres(text):
    text = text.lower()
    g = set()
for w, lst in MOOD_MAP.items():
    if w in text:
        g.update(lst)
    return list(g)

# ------------------------------

# NETFLIX FEATURES

# ------------------------------

def because_you_watched(watchlist, mdf, n=8):
    if not watchlist:
        return []

seed = watchlist[-1]
row = mdf[mdf['title']==seed]
if row.empty:
    return []

genres = row.iloc[0]['genres'].split('|')
recs = hybrid_recs(genres, watchlist, mdf.copy(), n)

return [(r['title'], r['genres'].split('|')[0]) for _, r in recs.iterrows()]

def trending_now(ratings_df, mdf, n=10):
    pop = ratings_df.groupby('movieId').size().sort_values(ascending=False).head(n)
    out = []
    for mid in pop.index:
        row = mdf[mdf['movieId']==mid]
        if not row.empty:
            t = row['title'].values[0]
            g = row['genres'].values[0].split('|')[0]
            out.append((t,g))
    return out

# ------------------------------

# CHATBOT

# ------------------------------

def chatbot_recommend(text, mdf, watchlist, n=8):
    g = mood_to_genres(text)
    if g:
        return hybrid_recs(g, watchlist, mdf.copy(), n)
return hybrid_recs(["Drama"], watchlist, mdf.copy(), n)

# ------------------------------

# UI

# ------------------------------

st.title("🎬 CineMatch")

# CHAT

st.subheader("Ask CineMatch")
q = st.text_input("Try: funny action or sad love story")

if q:
    recs = chatbot_recommend(q, movies_df, st.session_state.watchlist)
    st.dataframe(recs)

# GENRE PICK

genres = sorted({g for gs in movies_df['genres'] for g in gs.split('|')})
picked = st.multiselect("Pick Genres", genres)

if st.button("Recommend"):
    if picked:
        recs = hybrid_recs(picked, st.session_state.watchlist, movies_df.copy())
    st.dataframe(recs)

# NETFLIX SECTIONS

st.subheader("🔥 Trending Now")
trend = trending_now(ratings_df, movies_df)
st.write(trend)

if st.session_state.watchlist:
    st.subheader("Because You Watched")
    byw = because_you_watched(st.session_state.watchlist, movies_df)
    st.write(byw)
