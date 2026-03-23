import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, requests, urllib.parse

st.set_page_config(page_title="CineMatch", layout="wide")

# ───────────────────────────────────────────────────────────────
# ROUTING
# ───────────────────────────────────────────────────────────────
qp = st.query_params.to_dict()

if "movie" in qp:
    st.session_state["movie"] = urllib.parse.unquote(qp["movie"])
    st.session_state["page"] = "detail"
    st.query_params.clear()
    st.rerun()

if "nav" in qp:
    st.session_state["page"] = qp["nav"]
    st.session_state["movie"] = None
    st.query_params.clear()
    st.rerun()

for k, v in {
    "page": "home",
    "movie": None,
    "watchlist": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────
def nav_href(pg):
    return f"?nav={pg}"

def card_href(t):
    return f"?movie={urllib.parse.quote(t)}"

# ───────────────────────────────────────────────────────────────
# STYLES
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
body {background:#08090e; color:white;}
.nav {
  position:sticky; top:0;
  background:#08090e;
  padding:15px 50px;
  display:flex; align-items:center;
  border-bottom:1px solid rgba(255,255,255,0.08);
}
.nav-logo {
  font-family:serif;
  font-size:22px;
  letter-spacing:3px;
  color:#c9a96e;
  text-decoration:none;
  font-weight:bold;
}
.nav-links {
  margin-left:30px;
}
.nav-links a {
  margin-right:20px;
  color:#aaa;
  text-decoration:none;
}
.nav-links a:hover {color:#c9a96e;}
.nav-cta {
  margin-left:auto;
  background:#c9a96e;
  color:black;
  padding:8px 18px;
  border-radius:6px;
  text-decoration:none;
  font-weight:600;
}
.card {
  background:#141520;
  padding:10px;
  border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# NAVBAR
# ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="nav">
  <a class="nav-logo" href="{nav_href('home')}">CINEMATCH</a>

  <div class="nav-links">
    <a href="{nav_href('home')}">Browse</a>
    <a href="{nav_href('recs')}">For You</a>
    <a href="{nav_href('watchlist')}">Watchlist</a>
  </div>

  <a class="nav-cta" href="{nav_href('recs')}">🎯 Get Recommendations</a>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# DATA
# ───────────────────────────────────────────────────────────────
@st.cache_data
def load():
    movies = pd.read_csv("data/movies.csv")
    return movies

movies = load()

# ───────────────────────────────────────────────────────────────
# HOME PAGE
# ───────────────────────────────────────────────────────────────
if st.session_state.page == "home":

    st.title("Browse Movies")

    cols = st.columns(5)

    for i, row in movies.head(20).iterrows():
        with cols[i % 5]:
            st.markdown(f"""
<a href="{card_href(row['title'])}" target="_self">
  <div class="card">
    🎬 {row['title']}
  </div>
</a>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# DETAIL PAGE
# ───────────────────────────────────────────────────────────────
elif st.session_state.movie:

    title = st.session_state.movie
    st.title(title)

    if title not in st.session_state.watchlist:
        if st.button("Add to Watchlist"):
            st.session_state.watchlist.append(title)
            st.rerun()
    else:
        if st.button("Remove from Watchlist"):
            st.session_state.watchlist.remove(title)
            st.rerun()

# ───────────────────────────────────────────────────────────────
# WATCHLIST
# ───────────────────────────────────────────────────────────────
elif st.session_state.page == "watchlist":

    st.title("My Watchlist")

    for m in st.session_state.watchlist:
        st.markdown(f"""
<a href="{card_href(m)}" target="_self">
  🎬 {m}
</a>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ───────────────────────────────────────────────────────────────
elif st.session_state.page == "recs":

    st.title("Get Recommendations")

    query = st.text_input("What do you want to watch?")

    if query:
        tfidf = TfidfVectorizer()
        mat = tfidf.fit_transform(movies['genres'].fillna(''))

        q_vec = tfidf.transform([query])
        scores = cosine_similarity(q_vec, mat).flatten()

        top = scores.argsort()[::-1][:10]

        for i in top:
            st.markdown(f"""
<a href="{card_href(movies.iloc[i]['title'])}" target="_self">
  🎬 {movies.iloc[i]['title']}
</a>
""", unsafe_allow_html=True)
