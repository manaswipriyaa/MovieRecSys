import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, requests, urllib.parse

st.set_page_config(
    page_title="CineMatch",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Outfit:wght@300;400;500;600&display=swap');

:root {
  --bg:        #08090e;
  --surface:   #0e0f18;
  --surface2:  #141520;
  --border:    rgba(255,255,255,0.07);
  --accent:    #c9a96e;
  --accent2:   #e8c992;
  --accentdim: rgba(201,169,110,0.12);
  --red:       #e05c5c;
  --text:      #eaeaf5;
  --muted:     #5a5a72;
  --subtle:    #2a2a3e;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; -webkit-font-smoothing: antialiased; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stSidebar"], footer, header { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--subtle); border-radius: 3px; }

/* ── NAV ── */
.nav {
  height: 58px; background: rgba(8,9,14,0.94);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 44px; gap: 20px;
  position: sticky; top: 0; z-index: 1000;
}
.nav-logo {
  font-family: 'Playfair Display', serif; font-weight: 900;
  font-size: 1.1rem; letter-spacing: 5px; text-transform: uppercase; color: var(--accent);
}
.nav-pip { width: 3px; height: 3px; border-radius: 50%; background: var(--muted); }

div[data-testid="stHorizontalBlock"]:first-of-type {
  background: rgba(8,9,14,0.94) !important; backdrop-filter: blur(16px) !important;
  border-bottom: 1px solid var(--border) !important;
  position: sticky !important; top: 0 !important; z-index: 999 !important;
  margin-top: -59px !important; height: 58px !important;
  padding: 0 44px 0 256px !important;
  display: flex !important; align-items: center !important; gap: 2px !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] { flex: 0 0 auto !important; width: auto !important; padding: 0 !important; }
div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:last-child { flex: 1 1 auto !important; }
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
  background: transparent !important; color: var(--muted) !important;
  border: none !important; box-shadow: none !important;
  font-family: 'Outfit', sans-serif !important; font-size: 0.7rem !important;
  font-weight: 500 !important; letter-spacing: 2px !important;
  text-transform: uppercase !important; padding: 6px 18px !important;
  border-radius: 4px !important; width: auto !important; transform: none !important;
  transition: color 0.15s !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
  color: var(--accent) !important; background: transparent !important; transform: none !important;
}

/* ── HERO ── */
.hero {
  padding: 44px 44px 36px;
  background: linear-gradient(155deg, #0c0b1c 0%, #0f0c1f 55%, #080912 100%);
  position: relative; overflow: hidden;
}
.hero::before {
  content:''; position:absolute; top:-100px; right:-60px; width:400px; height:400px;
  border-radius:50%; pointer-events:none;
  background: radial-gradient(circle, rgba(201,169,110,0.07) 0%, transparent 65%);
}
.hero-eyebrow { font-size: 0.58rem; letter-spacing: 4px; text-transform: uppercase; color: var(--accent); font-weight: 500; margin-bottom: 12px; }
.hero-title {
  font-family: 'Playfair Display', serif; font-weight: 900;
  font-size: clamp(1.9rem, 3.8vw, 2.9rem); color: #fff;
  line-height: 1.1; margin-bottom: 12px; letter-spacing: -0.5px;
}
.hero-title em { font-style: italic; color: var(--accent); }
.hero-sub { font-size: 0.87rem; color: var(--muted); line-height: 1.75; max-width: 420px; font-weight: 300; }

/* ── GLOBAL BUTTONS ── */
.stButton > button {
  background: var(--accent) !important; color: #08090e !important;
  border: none !important; border-radius: 6px !important;
  font-family: 'Outfit', sans-serif !important; font-size: 0.7rem !important;
  font-weight: 600 !important; letter-spacing: 2px !important;
  text-transform: uppercase !important; padding: 10px 22px !important;
  box-shadow: 0 4px 16px rgba(201,169,110,0.18) !important;
  transition: opacity 0.15s !important; width: auto !important; transform: none !important;
}
.stButton > button:hover { opacity: 0.86 !important; transform: none !important; }

.cta-col .stButton > button { padding: 11px 26px !important; }
.wl-col .stButton > button {
  background: var(--surface) !important; color: var(--accent2) !important;
  border: 1px solid rgba(201,169,110,0.22) !important;
  box-shadow: none !important; padding: 10px 20px !important;
}
.wl-col .stButton > button:hover { border-color: rgba(201,169,110,0.5) !important; }

.back-btn .stButton > button {
  background: transparent !important; color: var(--muted) !important;
  border: 1px solid rgba(255,255,255,0.09) !important; box-shadow: none !important;
  font-size: 0.7rem !important; text-transform: none !important;
  letter-spacing: 0.5px !important; padding: 7px 18px !important;
}
.ghost-btn .stButton > button {
  background: transparent !important; color: var(--muted) !important;
  border: 1px solid rgba(255,255,255,0.09) !important; box-shadow: none !important;
}
.wl-remove .stButton > button {
  background: transparent !important; color: var(--muted) !important;
  border: 1px solid rgba(255,255,255,0.09) !important; box-shadow: none !important;
  font-size: 0.7rem !important;
}

/* ── DIVIDER ── */
.divider { height: 1px; background: var(--border); }

/* ── BROWSE ── */
.browse-wrap { padding: 32px 44px 0; }
.section-label { font-size: 0.58rem; letter-spacing: 3.5px; text-transform: uppercase; color: var(--muted); font-weight: 500; margin-bottom: 5px; }
.section-title { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.25rem; color: #fff; margin-bottom: 22px; letter-spacing: -0.3px; }
.filter-count { font-size: 0.62rem; color: var(--muted); letter-spacing: 2px; font-weight: 500; margin: 14px 0 18px; }

/* ── MOVIE CARD ── */
.mcard {
  border-radius: 8px; overflow: hidden; background: var(--surface);
  border: 1px solid var(--border); position: relative; cursor: pointer;
  transition: transform 0.22s cubic-bezier(.22,.68,0,1.2), box-shadow 0.22s, border-color 0.2s;
}
.mcard:hover {
  transform: translateY(-7px) scale(1.012);
  box-shadow: 0 20px 50px rgba(0,0,0,0.8), 0 0 0 1px rgba(201,169,110,0.22);
  border-color: rgba(201,169,110,0.25);
}
.mcard-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.mcard-ph { width: 100%; aspect-ratio: 2/3; background: linear-gradient(160deg, var(--surface), var(--surface2)); display: flex; align-items: center; justify-content: center; font-size: 2rem; color: var(--muted); }
.mcard-overlay {
  position: absolute; inset: 0;
  background: linear-gradient(to top, rgba(8,9,14,0.97) 0%, rgba(8,9,14,0.45) 38%, transparent 68%);
  opacity: 0; transition: opacity 0.22s;
  display: flex; flex-direction: column; justify-content: flex-end; padding: 12px 11px 11px;
}
.mcard:hover .mcard-overlay { opacity: 1; }
.mcard-ov-title { font-family:'Playfair Display',serif; font-size:0.75rem; font-weight:700; color:#fff; margin-bottom:3px; line-height:1.2; }
.mcard-ov-sub { font-size:0.6rem; color:var(--accent); letter-spacing:0.5px; }
.mcard-body { padding: 9px 10px 11px; }
.mcard-title { font-size:0.71rem; font-weight:500; color:#c0c0d8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:2px; }
.mcard-genre { font-size:0.59rem; color:var(--muted); }

/* Invisible full-card click — no visible button below poster */
.mcard-wrap { position: relative; }
.mcard-wrap .stButton { position: absolute; inset: 0; }
.mcard-wrap .stButton > button {
  position: absolute !important; inset: 0 !important;
  width: 100% !important; height: 100% !important;
  background: transparent !important; border: none !important;
  box-shadow: none !important; color: transparent !important;
  font-size: 0 !important; padding: 0 !important; cursor: pointer !important;
  border-radius: 8px !important; transform: none !important;
}
.mcard-wrap .stButton > button:hover { background: transparent !important; transform: none !important; }

/* ── BOTTOM BANNER ── */
.btm-banner {
  margin: 44px 44px 0;
  background: linear-gradient(120deg, #100e24, #17102a);
  border: 1px solid rgba(201,169,110,0.13); border-radius: 12px;
  padding: 34px 44px;
  display: flex; align-items: center; justify-content: space-between; gap: 24px;
  position: relative; overflow: hidden;
}
.btm-banner::after {
  content:''; position:absolute; right:-60px; top:-60px; width:260px; height:260px;
  border-radius:50%; background: radial-gradient(circle, rgba(201,169,110,0.07) 0%, transparent 65%);
}
.btm-title { font-family:'Playfair Display',serif; font-weight:700; font-size:1.3rem; color:#fff; margin-bottom:7px; }
.btm-sub { font-size:0.81rem; color:var(--muted); font-weight:300; }
.btm-banner .stButton > button { white-space: nowrap !important; }

/* ── DETAIL PAGE ── */
.detail-backdrop { width:100%; height:210px; position:relative; overflow:hidden; }
.detail-backdrop img { width:100%; height:100%; object-fit:cover; opacity:0.17; display:block; }
.detail-backdrop-fade { position:absolute; inset:0; background:linear-gradient(to top, var(--bg) 0%, transparent 55%); }
.detail-wrap { padding: 0 44px 56px; }
.detail-inner { display:flex; gap:38px; align-items:flex-start; margin-top:-76px; position:relative; z-index:2; }
.detail-poster { width:175px; min-width:175px; border-radius:10px; box-shadow:0 24px 64px rgba(0,0,0,0.88); display:block; }
.detail-poster-ph { width:175px; min-width:175px; height:262px; border-radius:10px; background:var(--surface2); display:flex; align-items:center; justify-content:center; font-size:2.8rem; }
.detail-info { flex:1; padding-top:84px; }
.detail-title { font-family:'Playfair Display',serif; font-weight:900; font-size:clamp(1.5rem,3vw,2.5rem); color:#fff; line-height:1.08; margin-bottom:12px; }
.detail-meta { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
.detail-rating { background:rgba(201,169,110,0.12); color:var(--accent2); font-size:0.73rem; font-weight:600; padding:4px 12px; border-radius:5px; border:1px solid rgba(201,169,110,0.22); }
.detail-year, .detail-runtime { font-size:0.77rem; color:var(--muted); }
.detail-tagline { font-style:italic; color:var(--muted); font-size:0.83rem; margin-bottom:14px; }
.detail-overview { font-size:0.87rem; color:#7878a0; line-height:1.82; max-width:600px; margin-bottom:20px; font-weight:300; }
.detail-pill { display:inline-block; background:rgba(255,255,255,0.04); color:#aaa; font-size:0.67rem; padding:3px 12px; border-radius:4px; border:1px solid rgba(255,255,255,0.07); margin-right:6px; margin-bottom:6px; }
.trailer-btn { display:inline-flex; align-items:center; gap:8px; background:rgba(224,92,92,0.1); color:#ef8080; font-size:0.73rem; font-weight:600; padding:10px 20px; border-radius:6px; border:1px solid rgba(224,92,92,0.2); text-decoration:none; transition:background 0.15s; }
.trailer-btn:hover { background:rgba(224,92,92,0.18); }
.sub-section { padding:28px 44px; border-top:1px solid var(--border); }
.sub-title { font-family:'Playfair Display',serif; font-weight:700; font-size:0.98rem; color:#fff; margin-bottom:18px; }
.providers { display:flex; flex-wrap:wrap; gap:11px; }
.provider { background:var(--surface); border-radius:10px; border:1px solid var(--border); padding:12px 15px; display:flex; flex-direction:column; align-items:center; gap:6px; min-width:88px; text-decoration:none; transition:border-color 0.15s, transform 0.15s; }
.provider:hover { border-color:rgba(201,169,110,0.32); transform:translateY(-2px); }
.provider img { width:42px; height:42px; border-radius:8px; object-fit:cover; }
.provider-name { font-size:0.62rem; color:#888; text-align:center; }
.provider-type { font-size:0.56rem; color:var(--muted); }
.no-stream { font-size:0.81rem; color:var(--muted); }
.cast-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(86px,1fr)); gap:9px; }
.cast-card { background:var(--surface); border-radius:7px; overflow:hidden; border:1px solid var(--border); transition:border-color 0.15s; }
.cast-card:hover { border-color:rgba(201,169,110,0.22); }
.cast-img { width:100%; aspect-ratio:2/3; object-fit:cover; display:block; }
.cast-ph { width:100%; aspect-ratio:2/3; background:var(--surface2); display:flex; align-items:center; justify-content:center; font-size:1.4rem; }
.cast-name { font-size:0.65rem; font-weight:600; color:#bbb; padding:6px 7px 2px; }
.cast-char { font-size:0.57rem; color:var(--muted); padding:0 7px 7px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* ── FOR YOU ── */
.foryou-wrap { padding: 36px 44px; }

/* rec card — whole card is the click target */
.rec-card { background:var(--surface); border-radius:10px; border:1px solid var(--border); padding:14px 16px; margin-bottom:10px; display:flex; gap:14px; align-items:flex-start; transition:border-color 0.18s, transform 0.18s, box-shadow 0.18s; cursor:pointer; }
.rec-card:hover { border-color:rgba(201,169,110,0.28); transform:translateX(4px); box-shadow:0 4px 24px rgba(0,0,0,0.4); }
.rec-num { font-family:'Playfair Display',serif; font-size:1.45rem; font-weight:900; color:rgba(201,169,110,0.14); min-width:36px; line-height:1; text-align:right; }
.rec-poster { width:50px; height:75px; border-radius:6px; object-fit:cover; flex-shrink:0; }
.rec-poster-ph { width:50px; height:75px; border-radius:6px; background:var(--surface2); display:flex; align-items:center; justify-content:center; font-size:1.2rem; flex-shrink:0; }
.rec-body { flex:1; min-width:0; }
.rec-title { font-size:0.87rem; font-weight:600; color:var(--text); margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.rec-pills { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:9px; }
.rec-pill { background:var(--accentdim); color:var(--accent2); font-size:0.59rem; padding:2px 8px; border-radius:3px; border:1px solid rgba(201,169,110,0.15); font-weight:500; }
.rec-bar-bg { height:2px; background:var(--subtle); border-radius:2px; margin-bottom:5px; }
.rec-bar { height:2px; background:linear-gradient(90deg,var(--accent),var(--accent2)); border-radius:2px; }
.rec-score { font-size:0.65rem; color:var(--accent); font-weight:500; }
.rec-hint { font-size:0.6rem; color:var(--muted); margin-top:3px; }

/* rec open-detail button — styled small */
.rec-open .stButton > button {
  background: transparent !important; color: var(--accent2) !important;
  border: 1px solid rgba(201,169,110,0.2) !important; box-shadow: none !important;
  font-size: 0.65rem !important; padding: 4px 14px !important; margin-top: 6px !important;
  text-transform: none !important; letter-spacing: 0.5px !important;
}
.rec-open .stButton > button:hover { background: var(--accentdim) !important; }

/* ── WATCHLIST PAGE ── */
.wl-empty { text-align:center; padding:80px 40px; color:var(--muted); }
.wl-empty-icon { font-size:2.8rem; margin-bottom:16px; }
.wl-empty-title { font-family:'Playfair Display',serif; font-size:1.2rem; color:#fff; margin-bottom:8px; }
.wl-empty-sub { font-size:0.83rem; }

/* ── INPUTS ── */
.stTextInput input { background:var(--surface) !important; color:var(--text) !important; border:1px solid rgba(255,255,255,0.09) !important; border-radius:7px !important; font-size:0.85rem !important; font-family:'Outfit',sans-serif !important; }
.stTextInput input:focus { border-color:rgba(201,169,110,0.4) !important; box-shadow:0 0 0 2px rgba(201,169,110,0.07) !important; }
div[data-baseweb="select"] > div { background:var(--surface) !important; border:1px solid rgba(255,255,255,0.09) !important; border-radius:7px !important; font-family:'Outfit',sans-serif !important; }
.stCheckbox label { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:6px !important; padding:5px 12px !important; font-size:0.7rem !important; font-weight:500 !important; color:var(--muted) !important; cursor:pointer !important; transition:border-color 0.15s,color 0.15s !important; white-space:nowrap !important; }
.stCheckbox label:hover { border-color:rgba(201,169,110,0.35) !important; color:var(--accent2) !important; }

.stDownloadButton > button { background:transparent !important; color:var(--muted) !important; border:1px solid rgba(255,255,255,0.09) !important; box-shadow:none !important; font-size:0.68rem !important; }
.stDownloadButton > button:hover { color:var(--text) !important; }

.chart-panel { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
TMDB_KEY = "e27947a52a97677530fdd9e476e8b17c"
IMG_BASE  = "https://image.tmdb.org/t/p"

GENRE_ICON = {
    'Action':'💥','Adventure':'🗺️','Animation':'🎨','Comedy':'😄',
    'Crime':'🕵️','Documentary':'📽️','Drama':'🎭','Fantasy':'🧙',
    'Horror':'👻','Musical':'🎵','Mystery':'🔍','Romance':'❤️',
    'Sci-Fi':'👽','Thriller':'🔪','War':'⚔️','Western':'🤠','Children':'🧒',
}

# Provider deep-link map
PROVIDER_LINKS = {
    'Netflix':            'https://www.netflix.com/search?q=',
    'Amazon Prime Video': 'https://www.amazon.com/s?k=',
    'Prime Video':        'https://www.amazon.com/s?k=',
    'Disney+':            'https://www.disneyplus.com/search/',
    'Hotstar':            'https://www.hotstar.com/in/search?q=',
    'Disney+ Hotstar':    'https://www.hotstar.com/in/search?q=',
    'Apple TV+':          'https://tv.apple.com/search?term=',
    'Hulu':               'https://www.hulu.com/search?q=',
    'HBO Max':            'https://www.max.com/search?q=',
    'Max':                'https://www.max.com/search?q=',
    'Peacock':            'https://www.peacocktv.com/search?q=',
    'Paramount+':         'https://www.paramountplus.com/search/',
    'Zee5':               'https://www.zee5.com/search/result/',
    'SonyLIV':            'https://www.sonyliv.com/search?q=',
    'Jio Cinema':         'https://www.jiocinema.com/search/',
}

def provider_href(name, movie_title, fallback=''):
    base = PROVIDER_LINKS.get(name, '')
    if base:
        return base + urllib.parse.quote(movie_title)
    return fallback or '#'

@st.cache_data(show_spinner=False)
def tmdb_search(title):
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params={"api_key": TMDB_KEY, "query": clean}, timeout=5)
        if r.status_code != 200:
            return None
        results = r.json().get('results', [])
        with_poster = [x for x in results if x.get('poster_path')]
        return with_poster[0] if with_poster else (results[0] if results else None)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def tmdb_details(title):
    sr = tmdb_search(title)
    if not sr:
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{sr['id']}",
            params={"api_key": TMDB_KEY,
                    "append_to_response": "credits,watch/providers,videos"},
            timeout=6
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def poster_url(title):
    r = tmdb_search(title)
    if r and r.get('poster_path'):
        return f"{IMG_BASE}/w300{r['poster_path']}"
    return None

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rp, mp = os.path.join("data","ratings.csv"), os.path.join("data","movies.csv")
    if not os.path.exists(rp) or not os.path.exists(mp):
        return None, None
    rd = pd.read_csv(rp)
    md = pd.read_csv(mp)
    md['genres'] = md['genres'].fillna('')
    return rd, md

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def genre_recs(picked, movies_df, n=12):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf.fit(movies_df['genres'])
    q = tfidf.transform(['|'.join(picked)])
    m = tfidf.transform(movies_df['genres'])
    cosine = cosine_similarity(q, m).flatten()
    def cov(g): return len(set(picked) & {x.strip() for x in g.split('|')}) / len(picked)
    coverage = movies_df['genres'].apply(cov).values
    combined = 0.6 * cosine + 0.4 * coverage
    mx = combined.max()
    if mx > 0: combined /= mx
    idx = combined.argsort()[::-1][:n]
    return pd.DataFrame([{'title': movies_df.iloc[i]['title'],
                           'genres': movies_df.iloc[i]['genres'],
                           'score': round(float(combined[i]), 4)} for i in idx])

def search_recs(query, movies_df, n=12):
    df = movies_df.copy()
    df['_t'] = df['title'].fillna('') + ' ' + df['genres'].fillna('').str.replace('|',' ')
    tfidf  = TfidfVectorizer(ngram_range=(1,2), max_features=30000)
    mat    = tfidf.fit_transform(df['_t'])
    scores = cosine_similarity(tfidf.transform([query]), mat).flatten()
    mx = scores.max()
    if mx > 0: scores /= mx
    idx = scores.argsort()[::-1][:n]
    return pd.DataFrame([{'title': df.iloc[i]['title'],
                           'genres': df.iloc[i]['genres'],
                           'score': round(float(scores[i]), 4)} for i in idx if scores[i] > 0])

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {'page':'home','prev':'home','movie':None,'genres':[],
              'recs':None,'watchlist':[],'rec_mode':'genre','search_q':''}.items():
    if k not in st.session_state:
        st.session_state[k] = v

ratings_df, movies_df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# NAV BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="nav"><div class="nav-logo">CineMatch</div><div class="nav-pip"></div></div>',
            unsafe_allow_html=True)

wl_count = len(st.session_state.watchlist)
c1, c2, c3, _sp = st.columns([1, 1.1, 1.6, 14])
with c1:
    if st.button("Browse", key="nav_browse"):
        st.session_state.update({'page':'home','movie':None}); st.rerun()
with c2:
    if st.button("For You", key="nav_foryou"):
        st.session_state.update({'page':'recs','movie':None}); st.rerun()
with c3:
    wl_lbl = f"Watchlist  ({wl_count})" if wl_count else "Watchlist"
    if st.button(wl_lbl, key="nav_wl"):
        st.session_state.update({'page':'watchlist','movie':None}); st.rerun()

if ratings_df is None:
    st.error("Data files not found. Add data/ratings.csv and data/movies.csv"); st.stop()

all_genres = sorted({g.strip() for gs in movies_df['genres']
                     for g in gs.split('|') if g.strip() not in ('','(no genres listed)')})

# ─────────────────────────────────────────────────────────────────────────────
# DETAIL PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_detail(title):
    st.markdown('<div style="padding:18px 44px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Back", key="back_btn"):
        st.session_state.movie = None; st.session_state.page = st.session_state.prev; st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

    det = tmdb_details(title)
    if not det or 'title' not in det:
        st.warning("Could not load details from TMDB."); return

    ptitle   = det.get('title', title)
    overview = det.get('overview','')
    tagline  = det.get('tagline','')
    year     = det.get('release_date','')[:4]
    rating   = round(det.get('vote_average',0),1)
    rt       = det.get('runtime') or 0
    runtime  = f"{rt//60}h {rt%60}m" if rt else ''
    genres   = [g['name'] for g in det.get('genres',[])]
    poster   = f"{IMG_BASE}/w400{det['poster_path']}"    if det.get('poster_path')   else None
    backdrop = f"{IMG_BASE}/w1280{det['backdrop_path']}" if det.get('backdrop_path') else None

    if backdrop:
        st.markdown(f'<div class="detail-backdrop"><img src="{backdrop}" alt=""/><div class="detail-backdrop-fade"></div></div>',
                    unsafe_allow_html=True)

    pimg = (f'<img class="detail-poster" src="{poster}" alt="{ptitle}" '
            f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
            f'<div class="detail-poster-ph" style="display:none;">🎬</div>'
            if poster else '<div class="detail-poster-ph">🎬</div>')
    gpills = ''.join(f'<span class="detail-pill">{g}</span>' for g in genres)

    st.markdown(f"""
    <div class="detail-wrap">
        <div class="detail-inner">
            {pimg}
            <div class="detail-info">
                <div class="detail-title">{ptitle}</div>
                <div class="detail-meta">
                    <span class="detail-year">{year}</span>
                    <span class="detail-rating">★ {rating} / 10</span>
                    <span class="detail-runtime">{runtime}</span>
                </div>
                {'<div class="detail-tagline">"'+tagline+'"</div>' if tagline else ''}
                <div class="detail-overview">{overview}</div>
                <div>{gpills}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Watchlist action
    in_wl = title in st.session_state.watchlist
    st.markdown('<div style="padding:0 44px 12px; display:flex; gap:10px;">', unsafe_allow_html=True)
    if in_wl:
        st.markdown('<div class="wl-remove">', unsafe_allow_html=True)
        if st.button("✓  In Watchlist — Remove", key="wl_tog"):
            st.session_state.watchlist.remove(title); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.button("＋  Add to Watchlist", key="wl_tog"):
            st.session_state.watchlist.append(title); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Trailer
    videos  = det.get('videos',{}).get('results',[])
    trailer = next((v for v in videos if v.get('type')=='Trailer' and v.get('site')=='YouTube'), None)
    if trailer:
        st.markdown(f'<div class="sub-section" style="border-top:none;padding-top:0;padding-bottom:18px;"><a class="trailer-btn" href="https://www.youtube.com/watch?v={trailer["key"]}" target="_blank">▶ &nbsp;Watch Trailer</a></div>',
                    unsafe_allow_html=True)

    # Where to Watch — clickable providers
    pd_data = det.get('watch/providers',{}).get('results',{})
    region  = pd_data.get('IN', pd_data.get('US',{}))
    jtw_link = region.get('link','')
    seen, combined_p = set(), []
    for label, lst in [('Stream', region.get('flatrate',[])),
                       ('Rent',   region.get('rent',[])),
                       ('Buy',    region.get('buy',[]))]:
        for p in lst:
            if p['provider_name'] not in seen:
                seen.add(p['provider_name']); combined_p.append((p, label))

    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Where to Watch</div>', unsafe_allow_html=True)
    if combined_p:
        html_p = ''
        for p, label in combined_p[:10]:
            logo = f'{IMG_BASE}/w92{p["logo_path"]}' if p.get('logo_path') else None
            img  = f'<img src="{logo}" alt="{p["provider_name"]}"/>' if logo \
                   else '<div style="width:42px;height:42px;background:var(--surface2);border-radius:8px;"></div>'
            href = provider_href(p['provider_name'], ptitle, jtw_link)
            html_p += f'<a class="provider" href="{href}" target="_blank" rel="noopener">{img}<div class="provider-name">{p["provider_name"]}</div><div class="provider-type">{label}</div></a>'
        st.markdown(f'<div class="providers">{html_p}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="no-stream">No streaming data available for your region.</p>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Cast
    cast = det.get('credits',{}).get('cast',[])[:14]
    if cast:
        ch = ''
        for c in cast:
            img = f'{IMG_BASE}/w185{c["profile_path"]}' if c.get('profile_path') else None
            ph  = f'<img class="cast-img" src="{img}" alt="{c["name"]}"/>' if img else '<div class="cast-ph">👤</div>'
            ch += f'<div class="cast-card">{ph}<div class="cast-name">{c["name"]}</div><div class="cast-char">{c.get("character","")[:22]}</div></div>'
        st.markdown(f'<div class="sub-section"><div class="sub-title">Cast</div><div class="cast-grid">{ch}</div></div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.movie:
    show_detail(st.session_state.movie); st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == 'home':

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">AI-Powered Discovery</div>
        <div class="hero-title">Your next favourite<br><em>film</em> awaits.</div>
        <div class="hero-sub">Browse thousands of movies or let our engine recommend
        films tailored exactly to your taste — no account needed.</div>
    </div>""", unsafe_allow_html=True)

    # CTA row + watchlist chip
    st.markdown('<div style="padding:20px 44px 0; display:flex; align-items:center; gap:14px; flex-wrap:wrap;">',
                unsafe_allow_html=True)
    ca, cb, _ = st.columns([1.3, 1.3, 8])
    with ca:
        st.markdown('<div class="cta-col">', unsafe_allow_html=True)
        if st.button("Get Recommendations →", key="hero_cta"):
            st.session_state.page = 'recs'; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        if st.session_state.watchlist:
            st.markdown('<div class="wl-col">', unsafe_allow_html=True)
            if st.button(f"🎯  Watchlist  ({wl_count})", key="wl_hero"):
                st.session_state.page = 'watchlist'; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider" style="margin-top:26px;"></div>', unsafe_allow_html=True)

    # Browse grid
    st.markdown('<div class="browse-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Explore</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Browse Movies</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        search = st.text_input("", placeholder="Search by title…",
                               label_visibility="collapsed", key="search_home")
    with fc2:
        gpick = st.selectbox("", ['All']+all_genres, label_visibility="collapsed", key="gpick")

    if search:
        pool = movies_df[movies_df['title'].str.contains(search, case=False, na=False)]
    elif gpick != 'All':
        pool = movies_df[movies_df['genres'].str.contains(gpick, case=False, na=False)]
    else:
        pool = movies_df.sample(min(40, len(movies_df)), random_state=42)
    filtered = pool.head(40)

    st.markdown(f'<div class="filter-count">{len(filtered)} TITLES</div>', unsafe_allow_html=True)

    cols = st.columns(8)
    for i, (_, row) in enumerate(filtered.iterrows()):
        t      = row['title']
        short  = t[:18]+'…' if len(t) > 18 else t
        genre1 = row['genres'].split('|')[0].strip() if row['genres'] else ''
        purl   = poster_url(t)
        icon   = GENRE_ICON.get(genre1, '🎬')

        with cols[i % 8]:
            img_html = (
                f'<img class="mcard-img" src="{purl}" loading="lazy" alt="{short}" '
                f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                f'<div class="mcard-ph" style="display:none;">{icon}</div>'
                if purl else f'<div class="mcard-ph">{icon}</div>'
            )
            st.markdown(f"""
            <div class="mcard">
                {img_html}
                <div class="mcard-overlay">
                    <div class="mcard-ov-title">{short}</div>
                    <div class="mcard-ov-sub">{genre1}</div>
                </div>
                <div class="mcard-body">
                    <div class="mcard-title">{short}</div>
                    <div class="mcard-genre">{genre1}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            # Invisible full-card click — no visible button under poster
            if st.button("", key=f"card_{i}", help=f"View {t}"):
                st.session_state.movie = t; st.session_state.prev = 'home'; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Bottom banner
    st.markdown('<div class="btm-banner">', unsafe_allow_html=True)
    st.markdown("""
        <div>
            <div class="btm-title">Not sure what to watch?</div>
            <div class="btm-sub">Tell us your favourite genres or describe what you're in the mood for.</div>
        </div>""", unsafe_allow_html=True)
    if st.button("Find My Movies →", key="banner_cta"):
        st.session_state.page = 'recs'; st.rerun()
    st.markdown('</div><div style="height:48px;"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == 'watchlist':
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Your Collection</div>
        <div class="hero-title">My <em>Watchlist</em></div>
        <div class="hero-sub">Films you've saved to watch later.</div>
    </div>""", unsafe_allow_html=True)

    wl = st.session_state.watchlist
    if not wl:
        st.markdown("""
        <div class="wl-empty">
            <div class="wl-empty-icon">🎬</div>
            <div class="wl-empty-title">Your watchlist is empty</div>
            <div class="wl-empty-sub">Open any movie and tap "Add to Watchlist".</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="browse-wrap">', unsafe_allow_html=True)
        st.markdown(f'<div class="filter-count">{len(wl)} SAVED</div>', unsafe_allow_html=True)
        cols = st.columns(8)
        for i, t in enumerate(wl):
            gr     = movies_df[movies_df['title'] == t]
            genre1 = gr['genres'].values[0].split('|')[0].strip() if len(gr) else ''
            purl   = poster_url(t)
            icon   = GENRE_ICON.get(genre1,'🎬')
            short  = t[:18]+'…' if len(t)>18 else t
            with cols[i % 8]:
                img_html = (
                    f'<img class="mcard-img" src="{purl}" loading="lazy" alt="{short}" '
                    f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                    f'<div class="mcard-ph" style="display:none;">{icon}</div>'
                    if purl else f'<div class="mcard-ph">{icon}</div>'
                )
                st.markdown(f"""
                <div class="mcard">
                    {img_html}
                    <div class="mcard-overlay">
                        <div class="mcard-ov-title">{short}</div>
                        <div class="mcard-ov-sub">{genre1}</div>
                    </div>
                    <div class="mcard-body">
                        <div class="mcard-title">{short}</div>
                        <div class="mcard-genre">{genre1}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
                if st.button("", key=f"wlcard_{i}", help=f"View {t}"):
                    st.session_state.movie = t; st.session_state.prev = 'watchlist'; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="padding:16px 44px 48px;">', unsafe_allow_html=True)
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("Clear Watchlist", key="clear_wl"):
            st.session_state.watchlist = []; st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOR YOU PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == 'recs':

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Personalised</div>
        <div class="hero-title">Made <em>For You</em></div>
        <div class="hero-sub">Pick genres, or describe what you're in the mood for.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="foryou-wrap">', unsafe_allow_html=True)

    # Mode toggle
    st.markdown('<div class="section-label" style="margin-bottom:12px;">Mode</div>', unsafe_allow_html=True)
    mc1, mc2, _ = st.columns([1, 1, 8])
    with mc1:
        if st.button("🎭  By Genre", key="mode_genre"):
            st.session_state.rec_mode = 'genre'; st.session_state.recs = None
    with mc2:
        if st.button("🔍  By Search", key="mode_search"):
            st.session_state.rec_mode = 'search'; st.session_state.recs = None

    mode = st.session_state.rec_mode
    st.markdown(f'<div style="font-size:0.65rem;color:var(--accent);margin:8px 0 26px;letter-spacing:2px;font-weight:600;">{"GENRE-BASED" if mode=="genre" else "SEARCH-BASED"}</div>',
                unsafe_allow_html=True)

    if mode == 'genre':
        st.markdown('<div class="section-label" style="margin-bottom:12px;">Select Genres</div>', unsafe_allow_html=True)
        gcols  = st.columns(9)
        picked = []
        for idx, g in enumerate(all_genres):
            icon = GENRE_ICON.get(g,'🎬')
            with gcols[idx % 9]:
                if st.checkbox(f"{icon} {g}", key=f"gc_{g}", value=g in st.session_state.genres):
                    picked.append(g)
        st.session_state.genres = picked  # persist

        if picked:
            st.markdown(f'<div style="font-size:0.66rem;color:var(--accent2);margin:12px 0 4px;letter-spacing:1px;font-weight:500;">SELECTED: {" · ".join(picked)}</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:22px;margin-bottom:10px;">Number of results</div>', unsafe_allow_html=True)
        top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn")
        b1, b2, _ = st.columns([1, 1, 8])
        with b1:
            find_btn = st.button("Find Movies", key="find_btn")
        with b2:
            st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
            if st.button("← Browse", key="back_b"):
                st.session_state.page = 'home'; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        if find_btn:
            if not picked: st.warning("Select at least one genre.")
            else:
                with st.spinner("Finding movies…"):
                    st.session_state.recs = genre_recs(picked, movies_df, top_n)
    else:
        st.markdown('<div class="section-label" style="margin-bottom:12px;">Describe what you want to watch</div>', unsafe_allow_html=True)
        sq = st.text_input("", placeholder="e.g.  'space adventure with humour'",
                           label_visibility="collapsed", key="sq_input",
                           value=st.session_state.search_q)
        st.session_state.search_q = sq
        top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn_s")
        b1, b2, _ = st.columns([1, 1, 8])
        with b1:
            srch_btn = st.button("Search", key="srch_btn")
        with b2:
            st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
            if st.button("← Browse", key="back_bs"):
                st.session_state.page = 'home'; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        if srch_btn:
            if not sq.strip(): st.warning("Enter a description.")
            else:
                with st.spinner("Searching…"):
                    st.session_state.recs = search_recs(sq, movies_df, top_n)

    # Results
    if st.session_state.recs is not None:
        recs = st.session_state.recs
        if recs.empty:
            st.warning("No results found. Try something different.")
        else:
            st.markdown('<div class="divider" style="margin:28px 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Your Results</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">Top {len(recs)} Picks For You</div>', unsafe_allow_html=True)

            lc, rc = st.columns([1.15, 1])
            with lc:
                for i, row in recs.iterrows():
                    bw    = int(row['score']*100)
                    pills = ''.join(f'<span class="rec-pill">{g.strip()}</span>'
                                    for g in row['genres'].split('|') if g.strip())
                    purl  = poster_url(row['title'])
                    g0    = row['genres'].split('|')[0].strip()
                    icon  = GENRE_ICON.get(g0,'🎬')
                    if purl:
                        ph_html = (f'<img class="rec-poster" src="{purl}" alt="poster" '
                                   f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                                   f'<div class="rec-poster-ph" style="display:none;">{icon}</div>')
                    else:
                        ph_html = f'<div class="rec-poster-ph">{icon}</div>'
                    num = f"0{i+1}" if i+1 < 10 else str(i+1)

                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-num">{num}</div>
                        {ph_html}
                        <div class="rec-body">
                            <div class="rec-title">{row['title']}</div>
                            <div class="rec-pills">{pills}</div>
                            <div class="rec-bar-bg"><div class="rec-bar" style="width:{bw}%;"></div></div>
                            <div class="rec-score">Match Score: {row['score']:.2f}</div>
                            <div class="rec-hint">Click below to open full details</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="rec-open">', unsafe_allow_html=True)
                    if st.button(f"Open Details — {row['title'][:28]}", key=f"rv_{i}"):
                        st.session_state.movie = row['title']
                        st.session_state.prev  = 'recs'; st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            with rc:
                st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, max(4, len(recs)*0.52)))
                fig.patch.set_facecolor('#0e0f18'); ax.set_facecolor('#0e0f18')
                titles = [r['title'][:26]+'…' if len(r['title'])>26 else r['title'] for _,r in recs.iterrows()]
                scores = recs['score'].values
                pal = ['#c9a96e','#cdb07c','#d1b78a','#d5be98','#d9c5a6',
                       '#ddcab0','#e1d0ba','#e5d5c4','#e9dace','#eddfd8']*2
                ax.barh(titles[::-1], scores[::-1], color=pal[:len(scores)][::-1],
                        height=0.5, edgecolor='none')
                for b, s in zip(ax.patches, scores[::-1]):
                    ax.text(b.get_width()+0.014, b.get_y()+b.get_height()/2,
                            f'{s:.2f}', va='center', color='#5a5a72', fontsize=7.5)
                ax.set_xlim(0, 1.22)
                ax.set_xlabel('Match Score', color='#3a3a52', fontsize=8)
                ax.set_title('Match Chart', color='#c9a96e', fontsize=9,
                             fontweight='bold', pad=10)
                ax.tick_params(colors='#3a3a52', labelsize=7)
                for s in ax.spines.values(): s.set_edgecolor('#1e1e2e')
                plt.tight_layout(pad=1.2)
                st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div style="margin-top:12px;">', unsafe_allow_html=True)
                st.download_button("⬇ Download CSV", recs.to_csv(index=False),
                                   "recommendations.csv","text/csv", key="dl_csv")
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
