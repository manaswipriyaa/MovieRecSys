import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, requests, urllib.parse, html

st.set_page_config(page_title="CineMatch", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────
for k, v in {"page": "home", "movie": None, "genres": [], "recs": None, 
             "watchlist": [], "rec_mode": "genre", "search_q": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# PROFESSIONAL CSS — CLEAN, NO DECORATIONS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg:      #0a0e27;
  --surf:    #121829;
  --surf2:   #1a1f3a;
  --bdr:     rgba(255,255,255,0.08);
  --accent:  #6366f1;
  --accent2: #818cf8;
  --txt:     #f1f5f9;
  --txt-sec: #94a3b8;
  --success: #10b981;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; color: var(--txt); }

/* Remove Streamlit defaults */
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stSidebar"], footer, header { display: none !important; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0 !important; }
.stVerticalBlock { gap: 0 !important; }
.stMainBlockContainer > div:first-child { margin-top: 0 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

/* ═══════════════════════════════════════
   NAVIGATION — Clean & Professional
═══════════════════════════════════════ */
.nav {
  position: sticky;
  top: 0;
  z-index: 9999;
  background: rgba(10,14,39,0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--bdr);
}
.nav-inner {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 60px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.nav-left { display: flex; align-items: center; gap: 32px; }
.nav-logo {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 1.1rem;
  letter-spacing: -0.5px;
  color: var(--accent);
  text-decoration: none;
  cursor: pointer;
  transition: color 0.2s;
}
.nav-logo:hover { color: var(--accent2); }
.nav-links {
  display: flex;
  gap: 8px;
}
.nav-link {
  font-family: 'Inter', sans-serif;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--txt-sec);
  text-decoration: none;
  padding: 8px 16px;
  border-radius: 6px;
  transition: all 0.2s;
  cursor: pointer;
  border: 1px solid transparent;
}
.nav-link:hover { color: var(--txt); background: rgba(99,102,241,0.1); }
.nav-link.active { color: var(--accent); border-color: var(--accent); }
.nav-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--accent);
  color: white;
  font-size: 0.7rem;
  font-weight: 700;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  margin-left: 8px;
}

/* ═══════════════════════════════════════
   BUTTONS — Clean, No Link Look
═══════════════════════════════════════ */
.stButton > button {
  background: var(--accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 600 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  padding: 11px 28px !important;
  box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
  width: auto !important;
  min-width: 0 !important;
  white-space: nowrap !important;
  transition: all 0.2s !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  background: var(--accent2) !important;
  box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
  transform: translateY(-2px) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
}

.btn-secondary .stButton > button {
  background: var(--surf2) !important;
  color: var(--txt-sec) !important;
  border: 1px solid var(--bdr) !important;
  box-shadow: none !important;
}
.btn-secondary .stButton > button:hover {
  background: rgba(99,102,241,0.1) !important;
  color: var(--txt) !important;
  border-color: var(--accent) !important;
  box-shadow: none !important;
}

.btn-danger .stButton > button {
  background: transparent !important;
  color: #ef4444 !important;
  border: 1px solid rgba(239,68,68,0.3) !important;
  box-shadow: none !important;
}
.btn-danger .stButton > button:hover {
  background: rgba(239,68,68,0.1) !important;
  border-color: rgba(239,68,68,0.6) !important;
}

.stDownloadButton > button {
  background: var(--surf2) !important;
  color: var(--txt-sec) !important;
  border: 1px solid var(--bdr) !important;
  box-shadow: none !important;
  font-size: 0.85rem !important;
  padding: 9px 18px !important;
}
.stDownloadButton > button:hover {
  background: rgba(99,102,241,0.1) !important;
  color: var(--txt) !important;
  border-color: var(--accent) !important;
}

/* ═══════════════════════════════════════
   HERO SECTION
═══════════════════════════════════════ */
.hero {
  background: linear-gradient(135deg, #0f1535 0%, #1a1f3a 50%, #0a0e27 100%);
  border-bottom: 1px solid var(--bdr);
  padding: 80px 60px;
}
.hero-content { max-width: 1400px; margin: 0 auto; }
.hero-tag {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 16px;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: clamp(2.2rem, 4vw, 3.5rem);
  line-height: 1.1;
  color: var(--txt);
  margin-bottom: 16px;
}
.hero-subtitle {
  font-size: 1rem;
  color: var(--txt-sec);
  max-width: 500px;
  line-height: 1.6;
  font-weight: 400;
}

/* ═══════════════════════════════════════
   CONTENT WRAPPER
═══════════════════════════════════════ */
.content { max-width: 1400px; margin: 0 auto; padding: 60px; }
.section-tag {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 12px;
}
.section-title {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 2rem;
  color: var(--txt);
  margin-bottom: 40px;
  letter-spacing: -0.5px;
}
.count-badge {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--txt-sec);
  margin: 24px 0 32px;
}

/* ═══════════════════════════════════════
   MOVIE GRID
═══════════════════════════════════════ */
.movie-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 20px;
  margin-bottom: 40px;
}
.movie-card {
  border-radius: 10px;
  overflow: hidden;
  background: var(--surf);
  border: 1px solid var(--bdr);
  transition: all 0.3s cubic-bezier(0.22, 0.68, 0, 1.2);
  cursor: pointer;
  display: flex;
  flex-direction: column;
  height: 100%;
}
.movie-card:hover {
  transform: translateY(-8px);
  border-color: var(--accent);
  box-shadow: 0 20px 40px rgba(99,102,241,0.2);
}
.movie-img {
  width: 100%;
  aspect-ratio: 2/3;
  object-fit: cover;
  display: block;
  background: var(--surf2);
}
.movie-img-placeholder {
  width: 100%;
  aspect-ratio: 2/3;
  background: linear-gradient(135deg, var(--surf2) 0%, var(--surf) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.5rem;
}
.movie-info {
  padding: 14px;
  flex: 1;
  display: flex;
  flex-direction: column;
}
.movie-title {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--txt);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 6px;
}
.movie-genre {
  font-size: 0.75rem;
  color: var(--txt-sec);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ═══════════════════════════════════════
   DETAIL PAGE
═══════════════════════════════════════ */
.detail-header {
  background: linear-gradient(135deg, #0f1535 0%, #1a1f3a 100%);
  padding: 40px 60px;
  border-bottom: 1px solid var(--bdr);
}
.detail-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 60px;
}
.detail-flex {
  display: flex;
  gap: 50px;
  align-items: flex-start;
  margin-bottom: 50px;
}
.detail-poster {
  width: 200px;
  min-width: 200px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  background: var(--surf2);
}
.detail-poster img {
  width: 100%;
  height: auto;
  display: block;
}
.detail-info { flex: 1; }
.detail-title {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: clamp(1.8rem, 3vw, 2.8rem);
  color: var(--txt);
  line-height: 1.2;
  margin-bottom: 16px;
}
.detail-meta {
  display: flex;
  gap: 16px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 20px;
}
.meta-badge {
  background: rgba(99,102,241,0.1);
  color: var(--accent);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 6px 14px;
  border-radius: 6px;
  border: 1px solid var(--accent);
}
.detail-desc {
  font-size: 0.95rem;
  color: var(--txt-sec);
  line-height: 1.8;
  max-width: 600px;
  margin-bottom: 24px;
}
.genre-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 24px;
}
.genre-pill {
  background: var(--surf2);
  color: var(--txt-sec);
  font-size: 0.8rem;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid var(--bdr);
}

.detail-section {
  border-top: 1px solid var(--bdr);
  padding: 40px 0;
}
.section-h {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1.3rem;
  color: var(--txt);
  margin-bottom: 24px;
}

.providers {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.provider-btn {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: 8px;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  text-decoration: none;
  transition: all 0.2s;
  cursor: pointer;
}
.provider-btn:hover {
  border-color: var(--accent);
  background: rgba(99,102,241,0.1);
}
.provider-img {
  width: 40px;
  height: 40px;
  border-radius: 6px;
  object-fit: cover;
}
.provider-name {
  font-size: 0.7rem;
  color: var(--txt-sec);
  text-align: center;
}
.provider-type {
  font-size: 0.65rem;
  color: var(--txt-sec);
}

.cast-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
}
.cast-card {
  background: var(--surf);
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--bdr);
  transition: all 0.2s;
}
.cast-card:hover {
  border-color: var(--accent);
}
.cast-img {
  width: 100%;
  aspect-ratio: 2/3;
  object-fit: cover;
  background: var(--surf2);
}
.cast-info {
  padding: 8px;
}
.cast-name {
  font-weight: 600;
  font-size: 0.75rem;
  color: var(--txt);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.cast-char {
  font-size: 0.7rem;
  color: var(--txt-sec);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ═══════════════════════════════════════
   RECOMMENDATIONS
═══════════════════════════════════════ */
.rec-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.rec-item {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 16px;
  display: flex;
  gap: 16px;
  align-items: flex-start;
  transition: all 0.2s;
  cursor: pointer;
}
.rec-item:hover {
  border-color: var(--accent);
  transform: translateX(4px);
  box-shadow: 0 8px 24px rgba(99,102,241,0.15);
}
.rec-num {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 1.6rem;
  color: rgba(99,102,241,0.2);
  min-width: 40px;
  text-align: center;
}
.rec-poster {
  width: 60px;
  height: 90px;
  border-radius: 6px;
  object-fit: cover;
  flex-shrink: 0;
  background: var(--surf2);
}
.rec-content { flex: 1; min-width: 0; }
.rec-title {
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--txt);
  margin-bottom: 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.rec-genres {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 8px;
}
.rec-genre-tag {
  background: rgba(99,102,241,0.1);
  color: var(--accent);
  font-size: 0.7rem;
  padding: 2px 6px;
  border-radius: 3px;
  border: 1px solid rgba(99,102,241,0.2);
}
.rec-score {
  font-size: 0.8rem;
  color: var(--accent);
  font-weight: 600;
}

/* ═══════════════════════════════════════
   EMPTY STATE
═══════════════════════════════════════ */
.empty-state {
  text-align: center;
  padding: 80px 40px;
  color: var(--txt-sec);
}
.empty-icon {
  font-size: 3.5rem;
  margin-bottom: 20px;
}
.empty-title {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1.3rem;
  color: var(--txt);
  margin-bottom: 8px;
}
.empty-text {
  font-size: 0.95rem;
  max-width: 400px;
  margin: 0 auto;
}

/* ═══════════════════════════════════════
   INPUTS
═══════════════════════════════════════ */
.stTextInput input {
  background: var(--surf) !important;
  color: var(--txt) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 8px !important;
  font-size: 0.9rem !important;
  padding: 11px 14px !important;
  transition: all 0.2s !important;
}
.stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
.stTextInput input::placeholder {
  color: var(--txt-sec) !important;
}
div[data-baseweb="select"] > div {
  background: var(--surf) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 8px !important;
  color: var(--txt) !important;
}
.stCheckbox label {
  background: var(--surf) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 6px !important;
  padding: 8px 12px !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  color: var(--txt-sec) !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  white-space: nowrap !important;
  margin-bottom: 8px !important;
}
.stCheckbox label:hover {
  border-color: var(--accent) !important;
  background: rgba(99,102,241,0.05) !important;
  color: var(--txt) !important;
}

/* ═══════════════════════════════════════
   UTILITY
═══════════════════════════════════════ */
.divider { height: 1px; background: var(--bdr); margin: 40px 0; }
.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.5);
  backdrop-filter: blur(4px);
  z-index: 999;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
TMDB_KEY = "e27947a52a97677530fdd9e476e8b17c"
IMG_BASE = "https://image.tmdb.org/t/p"

GENRE_ICON = {
    'Action':'💥', 'Adventure':'🗺️', 'Animation':'🎨', 'Comedy':'😄',
    'Crime':'🕵️', 'Documentary':'📽️', 'Drama':'🎭', 'Fantasy':'🧙',
    'Horror':'👻', 'Musical':'🎵', 'Mystery':'🔍', 'Romance':'❤️',
    'Sci-Fi':'👽', 'Thriller':'🔪', 'War':'⚔️', 'Western':'🤠', 'Children':'🧒',
}

PROV_MAP = {
    'Netflix':'https://www.netflix.com/search?q=',
    'Amazon Prime Video':'https://www.amazon.com/s?k=',
    'Prime Video':'https://www.amazon.com/s?k=',
    'Disney+':'https://www.disneyplus.com/search/',
    'Hotstar':'https://www.hotstar.com/in/search?q=',
    'Disney+ Hotstar':'https://www.hotstar.com/in/search?q=',
    'Apple TV+':'https://tv.apple.com/search?term=',
    'Hulu':'https://www.hulu.com/search?q=',
    'Max':'https://www.max.com/search?q=',
    'HBO Max':'https://www.max.com/search?q=',
    'Peacock':'https://www.peacocktv.com/search?q=',
    'Paramount+':'https://www.paramountplus.com/search/',
    'Zee5':'https://www.zee5.com/search/result/',
    'SonyLIV':'https://www.sonyliv.com/search?q=',
    'Jio Cinema':'https://www.jiocinema.com/search/',
}

def prov_href(name, title, fb=''):
    b = PROV_MAP.get(name, '')
    return (b + urllib.parse.quote(title)) if b else (fb or '#')

# ─────────────────────────────────────────────────────────────────
# TMDB API
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def tmdb_search(title):
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params={"api_key": TMDB_KEY, "query": clean}, timeout=5)
        res = r.json().get('results', []) if r.status_code == 200 else []
        wp = [x for x in res if x.get('poster_path')]
        return wp[0] if wp else (res[0] if res else None)
    except: return None

@st.cache_data(show_spinner=False)
def tmdb_details(title):
    sr = tmdb_search(title)
    if not sr: return None
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{sr['id']}",
                         params={"api_key": TMDB_KEY,
                                 "append_to_response": "credits,watch/providers,videos"},
                         timeout=6)
        return r.json() if r.status_code == 200 else None
    except: return None

@st.cache_data(show_spinner=False)
def poster_url(title):
    r = tmdb_search(title)
    return f"{IMG_BASE}/w300{r['poster_path']}" if r and r.get('poster_path') else None

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rp = os.path.join("data", "ratings.csv")
    mp = os.path.join("data", "movies.csv")
    if not os.path.exists(rp) or not os.path.exists(mp): return None, None
    rd = pd.read_csv(rp)
    md = pd.read_csv(mp); md['genres'] = md['genres'].fillna('')
    return rd, md

ratings_df, movies_df = load_data()

all_genres = []
if movies_df is not None:
    all_genres = sorted({g.strip() for gs in movies_df['genres']
                         for g in gs.split('|')
                         if g.strip() not in ('', '(no genres listed)')})

if ratings_df is None:
    st.error("Data files not found. Add data/ratings.csv and data/movies.csv.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────
def genre_recs(picked, movies_df, n=12):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf.fit(movies_df['genres'])
    cos = cosine_similarity(tfidf.transform(['|'.join(picked)]),
                            tfidf.transform(movies_df['genres'])).flatten()
    def cov(g): return len(set(picked) & {x.strip() for x in g.split('|')}) / len(picked)
    comb = 0.6 * cos + 0.4 * movies_df['genres'].apply(cov).values
    mx = comb.max()
    if mx > 0: comb /= mx
    idx = comb.argsort()[::-1][:n]
    return pd.DataFrame([{'title': movies_df.iloc[i]['title'],
                           'genres': movies_df.iloc[i]['genres'],
                           'score': round(float(comb[i]), 4)} for i in idx])

def search_recs(query, movies_df, n=12):
    df = movies_df.copy()
    df['_t'] = df['title'].fillna('') + ' ' + df['genres'].fillna('').str.replace('|', ' ')
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    mat = tfidf.fit_transform(df['_t'])
    sc = cosine_similarity(tfidf.transform([query]), mat).flatten()
    mx = sc.max()
    if mx > 0: sc /= mx
    idx = sc.argsort()[::-1][:n]
    return pd.DataFrame([{'title': df.iloc[i]['title'],
                           'genres': df.iloc[i]['genres'],
                           'score': round(float(sc[i]), 4)} for i in idx if sc[i] > 0])

# ─────────────────────────────────────────────────────────────────
# NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────
p = st.session_state.page
wlc = len(st.session_state.watchlist)
badge = f'<span class="nav-badge">{wlc}</span>' if wlc else ''

def nc(page_id):
    return "nav-link active" if p == page_id else "nav-link"

st.markdown(f"""
<div class="nav">
  <div class="nav-inner">
    <div class="nav-left">
      <a class="nav-logo" onclick="location.href='?page=home'; window.location.reload();">CineMatch</a>
    </div>
    <div class="nav-links">
      <a class="{nc('home')}" onclick="window.location.href='?page=home'">BROWSE</a>
      <a class="{nc('recs')}" onclick="window.location.href='?page=recs'">FOR YOU</a>
      <a class="{nc('watchlist')}" onclick="window.location.href='?page=watchlist'">WATCHLIST{badge}</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Handle page navigation via query params
qp = st.query_params.to_dict()
if "page" in qp and qp["page"] in ["home", "recs", "watchlist"]:
    st.session_state.page = qp["page"]
    st.query_params.clear()

# ─────────────────────────────────────────────────────────────────
# RENDER GRID
# ─────────────────────────────────────────────────────────────────
def render_movie_grid(items):
    grid_html = '<div class="movie-grid">'
    for title, genre in items:
        purl = poster_url(title)
        icon = GENRE_ICON.get(genre, '🎬')
        title_esc = html.escape(title)
        genre_esc = html.escape(genre)
        short = title[:16] + '…' if len(title) > 16 else title
        short_esc = html.escape(short)
        
        img = (f'<img class="movie-img" src="{purl}" alt="{title_esc}" loading="lazy" '
               f'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\'"/>'
               f'<div class="movie-img-placeholder" style="display:none;">{icon}</div>'
               if purl else f'<div class="movie-img-placeholder">{icon}</div>')
        
        grid_html += f"""
<div class="movie-card" onclick="openMovie('{urllib.parse.quote(title)}')">
  {img}
  <div class="movie-info">
    <div class="movie-title" title="{title_esc}">{short_esc}</div>
    <div class="movie-genre">{genre_esc}</div>
  </div>
</div>"""
    
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)
    st.markdown(f"""
<script>
function openMovie(title) {{
    st.session_state.movie = decodeURIComponent(title);
    st.rerun();
}}
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────
if st.session_state.page == 'home':
    st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div class="hero-tag">AI-Powered Discovery</div>
    <div class="hero-title">Your next favourite film awaits.</div>
    <div class="hero-subtitle">Browse thousands of movies or let our engine recommend films tailored to your taste — no account needed.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 1.8, 4])
    with col1:
        if st.button("🎯 Get Recommendations", key="cta1"):
            st.session_state.page = 'recs'
            st.rerun()
    with col2:
        if st.session_state.watchlist:
            st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
            if st.button(f"My Watchlist ({wlc})", key="wl_home"):
                st.session_state.page = 'watchlist'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-tag">Explore</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Browse Movies</div>', unsafe_allow_html=True)
    
    f1, f2 = st.columns([3, 1])
    with f1:
        search = st.text_input("", placeholder="Search by title…", label_visibility="collapsed", key="search_home")
    with f2:
        gpick = st.selectbox("", ['All'] + all_genres, label_visibility="collapsed", key="gpick")

    if search:
        pool = movies_df[movies_df['title'].str.contains(search, case=False, na=False)]
    elif gpick != 'All':
        pool = movies_df[movies_df['genres'].str.contains(gpick, case=False, na=False)]
    else:
        pool = movies_df.sample(min(40, len(movies_df)), random_state=42)
    
    filtered = pool.head(40)
    st.markdown(f'<div class="count-badge">{len(filtered)} TITLES</div>', unsafe_allow_html=True)
    
    items = [(row['title'], row['genres'].split('|')[0].strip() if row['genres'] else 'Unknown')
             for _, row in filtered.iterrows()]
    render_movie_grid(items)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom CTA
    st.markdown("""
<div style="background: linear-gradient(135deg, #0f1535 0%, #1a1f3a 100%); border-top: 1px solid rgba(255,255,255,0.08); padding: 60px; margin-top: 40px;">
  <div style="max-width: 1400px; margin: 0 auto;">
    <h2 style="font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.8rem; color: white; margin-bottom: 12px;">
      Not sure what to watch?
    </h2>
    <p style="color: rgba(241, 245, 249, 0.6); font-size: 1rem; margin-bottom: 24px; max-width: 500px;">
      Tell us your favourite genres or describe what you're in the mood for, and let our AI do the thinking.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 4, 4])
    with col1:
        if st.button("Discover Now →", key="cta_bottom"):
            st.session_state.page = 'recs'
            st.rerun()

# ─────────────────────────────────────────────────────────────────
# WATCHLIST PAGE
# ─────────────────────────────────────────────────────────────────
elif st.session_state.page == 'watchlist':
    st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div class="hero-tag">Your Collection</div>
    <div class="hero-title">My Watchlist</div>
    <div class="hero-subtitle">Films you've saved to watch later.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    wl = st.session_state.watchlist
    if not wl:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">🍿</div>
  <div class="empty-title">Your watchlist is empty</div>
  <div class="empty-text">Open any movie and tap "Add to Watchlist" to get started.</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="content">', unsafe_allow_html=True)
        st.markdown(f'<div class="count-badge">{len(wl)} SAVED</div>', unsafe_allow_html=True)
        
        wl_items = []
        for t in wl:
            gr = movies_df[movies_df['title'] == t]
            genre1 = gr['genres'].values[0].split('|')[0].strip() if len(gr) else 'Unknown'
            wl_items.append((t, genre1))
        
        render_movie_grid(wl_items)
        
        col1, col2, col3 = st.columns([1.5, 4, 4])
        with col1:
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            if st.button("Clear Watchlist", key="clear_wl"):
                st.session_state.watchlist = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# RECOMMENDATIONS PAGE
# ─────────────────────────────────────────────────────────────────
elif st.session_state.page == 'recs':
    st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div class="hero-tag">Personalised</div>
    <div class="hero-title">Made For You</div>
    <div class="hero-subtitle">Pick genres, or describe what you're in the mood for.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-tag">Recommendation Mode</div>', unsafe_allow_html=True)
    mc1, mc2, _ = st.columns([1.2, 1.5, 6])
    with mc1:
        if st.button("🎭 By Genre", key="mode_genre"):
            st.session_state.rec_mode = 'genre'
            st.session_state.recs = None
            st.rerun()
    with mc2:
        st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
        if st.button("🔍 By Search", key="mode_search"):
            st.session_state.rec_mode = 'search'
            st.session_state.recs = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    mode = st.session_state.rec_mode
    
    # GENRE MODE
    if mode == 'genre':
        st.markdown('<div class="section-tag">Select genres you love</div>', unsafe_allow_html=True)
        gcols = st.columns(6)
        picked = []
        for idx, g in enumerate(all_genres):
            with gcols[idx % 6]:
                if st.checkbox(f"{GENRE_ICON.get(g, '🎬')} {g}", key=f"gc_{g}",
                               value=g in st.session_state.genres):
                    picked.append(g)
        st.session_state.genres = picked
        
        if picked:
            st.markdown(f'<p style="color: var(--accent); font-weight: 600; margin: 20px 0;">✓ Selected: {", ".join(picked)}</p>', 
                       unsafe_allow_html=True)
        
        st.markdown('<div class="section-tag" style="margin-top: 32px;">Number of results</div>', unsafe_allow_html=True)
        top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn")
        
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.2, 1.5, 6])
        with col1:
            find_btn = st.button("Find Movies", key="find_btn")
        with col2:
            st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
            if st.button("← Back", key="back_genre"):
                st.session_state.page = 'home'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        if find_btn:
            if not picked:
                st.warning("Please select at least one genre.")
            else:
                with st.spinner("Finding movies…"):
                    st.session_state.recs = genre_recs(picked, movies_df, top_n)
    
    # SEARCH MODE
    else:
        st.markdown('<div class="section-tag">Describe what you want to watch</div>', unsafe_allow_html=True)
        sq = st.text_input("", placeholder="e.g. 'sci-fi adventure with humor'", label_visibility="collapsed",
                          key="sq_input", value=st.session_state.search_q)
        st.session_state.search_q = sq
        
        st.markdown('<div class="section-tag" style="margin-top: 32px;">Number of results</div>', unsafe_allow_html=True)
        top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn_s")
        
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.3, 1.5, 6])
        with col1:
            srch_btn = st.button("Search Movies", key="srch_btn")
        with col2:
            st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
            if st.button("← Back", key="back_search"):
                st.session_state.page = 'home'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        if srch_btn:
            if not sq.strip():
                st.warning("Please enter a description.")
            else:
                with st.spinner("Searching…"):
                    st.session_state.recs = search_recs(sq, movies_df, top_n)
    
    # RESULTS
    if st.session_state.recs is not None:
        recs = st.session_state.recs
        if recs.empty:
            st.warning("No results found. Try something different.")
        else:
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="content">', unsafe_allow_html=True)
            st.markdown('<div class="section-tag">Results</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">Top {len(recs)} Picks For You</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1.3, 1])
            
            with col1:
                st.markdown('<div class="rec-list">', unsafe_allow_html=True)
                for i, row in recs.iterrows():
                    purl = poster_url(row['title'])
                    g0 = row['genres'].split('|')[0].strip()
                    icon = GENRE_ICON.get(g0, '🎬')
                    title_esc = html.escape(row['title'])
                    
                    genres_html = ''.join(f'<span class="rec-genre-tag">{html.escape(g.strip())}</span>'
                                         for g in row['genres'].split('|') if g.strip())
                    
                    img = f'<img class="rec-poster" src="{purl}" alt="poster"/>' if purl else f'<div style="width:60px;height:90px;background:var(--surf2);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;">{icon}</div>'
                    
                    st.markdown(f"""
<div class="rec-item" onclick="openMovie('{urllib.parse.quote(row['title'])}')">
    <div class="rec-num">{i+1:02d}</div>
    {img}
    <div class="rec-content">
        <div class="rec-title">{title_esc}</div>
        <div class="rec-genres">{genres_html}</div>
        <div class="rec-score">Match: {row['score']:.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(5, max(4, len(recs) * 0.45)))
                fig.patch.set_facecolor('#121829')
                ax.set_facecolor('#121829')
                titles = [r['title'][:20] + '…' if len(r['title']) > 20 else r['title'] for _, r in recs.iterrows()]
                scores = recs['score'].values
                colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(scores)))
                ax.barh(titles[::-1], scores[::-1], color=colors[::-1], height=0.6, edgecolor='none')
                ax.set_xlim(0, 1.1)
                ax.set_xlabel('Match Score', color='#94a3b8', fontsize=9)
                ax.tick_params(colors='#94a3b8', labelsize=8)
                for sp in ax.spines.values():
                    sp.set_edgecolor('#1a1f3a')
                plt.tight_layout(pad=1)
                st.pyplot(fig, use_container_width=True)
                plt.close()
                
                st.download_button("⬇ Download CSV", recs.to_csv(index=False),
                                 "recommendations.csv", "text/csv", key="dl_csv")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MOVIE DETAIL PAGE (Overlay)
# ─────────────────────────────────────────────────────────────────
if st.session_state.movie:
    title = st.session_state.movie
    det = tmdb_details(title)
    
    if det and 'title' in det:
        st.markdown('<div class="detail-header">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("← Back", key="detail_back"):
                st.session_state.movie = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="detail-container">', unsafe_allow_html=True)
        
        ptitle = html.escape(str(det.get('title', title) or title))
        overview = html.escape(str(det.get('overview', '') or ''))
        year = str(det.get('release_date', '')[:4])
        rating = round(det.get('vote_average', 0), 1)
        runtime = det.get('runtime') or 0
        runtime_str = f"{runtime//60}h {runtime%60}m" if runtime else ''
        genres = [html.escape(str(g['name'])) for g in det.get('genres', [])]
        poster = f"{IMG_BASE}/w400{det['poster_path']}" if det.get('poster_path') else None
        
        st.markdown('<div class="detail-flex">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([0.15, 1])
        with col1:
            if poster:
                st.markdown(f'<div class="detail-poster"><img src="{poster}" alt="{ptitle}"/></div>',
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="detail-poster" style="display:flex;align-items:center;justify-content:center;font-size:3rem;">🎬</div>',
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
<div class="detail-info">
    <div class="detail-title">{ptitle}</div>
    <div class="detail-meta">
        <span class="meta-badge">⭐ {rating} / 10</span>
        <span class="meta-badge">{year}</span>
        <span class="meta-badge">{runtime_str}</span>
    </div>
    <div class="detail-desc">{overview}</div>
    <div class="genre-pills">
        {''.join(f'<span class="genre-pill">{g}</span>' for g in genres)}
    </div>
</div>
""", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Watchlist button
        in_wl = title in st.session_state.watchlist
        st.markdown('<div style="margin: 32px 0;">', unsafe_allow_html=True)
        if in_wl:
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            if st.button("✓ In Watchlist — Remove", key="wl_remove"):
                st.session_state.watchlist.remove(title)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.button("+ Add to Watchlist", key="wl_add"):
                st.session_state.watchlist.append(title)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Trailer
        videos = det.get('videos', {}).get('results', [])
        trailer = next((v for v in videos if v.get('type') == 'Trailer' and v.get('site') == 'YouTube'), None)
        if trailer:
            st.markdown(f'<a href="https://www.youtube.com/watch?v={trailer["key"]}" target="_blank" style="display:inline-flex;align-items:center;gap:8px;background:rgba(239,68,68,0.1);color:#ef4444;padding:12px 20px;border-radius:8px;text-decoration:none;border:1px solid rgba(239,68,68,0.3);">▶ Watch Trailer</a>',
                       unsafe_allow_html=True)
        
        # Where to watch
        st.markdown('<div class="detail-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-h">Where to Watch</div>', unsafe_allow_html=True)
        
        pd_data = det.get('watch/providers', {}).get('results', {})
        region = pd_data.get('IN', pd_data.get('US', {}))
        seen, combp = set(), []
        for lbl, lst in [('Stream', region.get('flatrate', [])),
                         ('Rent', region.get('rent', [])),
                         ('Buy', region.get('buy', []))]:
            for p2 in lst:
                if p2['provider_name'] not in seen:
                    seen.add(p2['provider_name'])
                    combp.append((p2, lbl))
        
        if combp:
            st.markdown('<div class="providers">', unsafe_allow_html=True)
            for p2, lbl in combp[:8]:
                logo = f'{IMG_BASE}/w92{p2["logo_path"]}' if p2.get('logo_path') else None
                href = prov_href(p2['provider_name'], ptitle)
                img = f'<img class="provider-img" src="{logo}" alt="{html.escape(p2["provider_name"])}"/>' if logo else f'<div class="provider-img" style="background:var(--surf2);"></div>'
                st.markdown(f"""
<a class="provider-btn" href="{href}" target="_blank" rel="noopener">
    {img}
    <span class="provider-name">{html.escape(p2["provider_name"])}</span>
    <span class="provider-type">{lbl}</span>
</a>
""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--txt-sec);">No streaming data available for your region.</p>',
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cast
        cast = det.get('credits', {}).get('cast', [])[:12]
        if cast:
            st.markdown('<div class="detail-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-h">Cast</div>', unsafe_allow_html=True)
            st.markdown('<div class="cast-grid">', unsafe_allow_html=True)
            for c in cast:
                cname = html.escape(str(c.get('name', '') or ''))
                cchar = html.escape(str(c.get('character', '') or '')[:20])
                img = f'{IMG_BASE}/w185{c["profile_path"]}' if c.get('profile_path') else None
                
                if img:
                    st.markdown(f"""
<div class="cast-card">
    <img class="cast-img" src="{img}" alt=""/>
    <div class="cast-info">
        <div class="cast-name">{cname}</div>
        <div class="cast-char">{cchar}</div>
    </div>
</div>
""", unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
