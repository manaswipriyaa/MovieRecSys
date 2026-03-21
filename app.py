import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, requests

st.set_page_config(
    page_title="MovieRecSys",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# STYLES  (overhauled)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:       #050609;
  --surface:  #0c0d14;
  --surface2: #13141f;
  --border:   rgba(255,255,255,0.06);
  --accent:   #6c63ff;
  --accent2:  #a78bfa;
  --pink:     #f472b6;
  --text:     #e2e2ee;
  --muted:    #55556a;
  --dim:      #2a2a3a;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stSidebar"], footer, header { display: none !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--dim); border-radius: 3px; }

/* ── NAV ── */
.nav {
  height: 60px;
  background: rgba(5,6,9,0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center;
  padding: 0 40px; gap: 28px;
  position: sticky; top: 0; z-index: 1000;
}
.nav-logo {
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: 1.25rem; letter-spacing: 3px; text-transform: uppercase;
  background: linear-gradient(90deg, #6c63ff, #a78bfa, #f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; flex-shrink: 0;
}
.nav-divider { width: 1px; height: 18px; background: var(--border); }

/* push streamlit nav buttons into the sticky bar */
div[data-testid="stHorizontalBlock"]:first-of-type {
  background: rgba(5,6,9,0.85) !important;
  backdrop-filter: blur(12px) !important;
  border-bottom: 1px solid var(--border) !important;
  position: sticky !important; top: 0 !important; z-index: 999 !important;
  margin-top: -61px !important;
  height: 60px !important;
  padding: 0 40px 0 270px !important;
  display: flex !important; align-items: center !important; gap: 2px !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] {
  flex: 0 0 auto !important; width: auto !important; padding: 0 !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:last-child {
  flex: 1 1 auto !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
  background: transparent !important; color: var(--muted) !important;
  border: none !important; box-shadow: none !important;
  font-size: 0.76rem !important; font-weight: 600 !important;
  letter-spacing: 1.5px !important; text-transform: uppercase !important;
  padding: 6px 16px !important; border-radius: 6px !important;
  width: auto !important; transition: all 0.15s !important; transform: none !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
  color: var(--text) !important; background: rgba(255,255,255,0.05) !important;
  transform: none !important;
}

/* ── HERO ── */
.hero {
  width: 100%; padding: 80px 40px 68px;
  background: linear-gradient(150deg, #08071a 0%, #0d0820 55%, #060812 100%);
  position: relative; overflow: hidden;
}
.hero-glow-1 {
  position: absolute; top: -120px; right: -80px;
  width: 520px; height: 520px; border-radius: 50%;
  background: radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 65%);
  pointer-events: none;
}
.hero-glow-2 {
  position: absolute; bottom: -80px; left: 5%;
  width: 380px; height: 380px; border-radius: 50%;
  background: radial-gradient(circle, rgba(244,114,182,0.07) 0%, transparent 65%);
  pointer-events: none;
}
.hero-label {
  font-size: 0.62rem; letter-spacing: 4px; text-transform: uppercase;
  color: var(--accent2); font-weight: 600; margin-bottom: 20px;
  display: flex; align-items: center; gap: 10px;
}
.hero-label::before {
  content: ''; width: 24px; height: 1px; background: var(--accent2);
}
.hero-title {
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: clamp(2.8rem, 5vw, 4.2rem); letter-spacing: -1px;
  color: #fff; line-height: 1.05; margin-bottom: 20px;
}
.hero-title em {
  font-style: normal;
  background: linear-gradient(90deg, #6c63ff, #a78bfa, #f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
  font-size: 0.95rem; color: var(--muted); line-height: 1.8;
  max-width: 500px; font-weight: 300;
}
.hero-stats {
  display: flex; gap: 32px; margin-top: 36px;
}
.hero-stat-num {
  font-family: 'Syne', sans-serif; font-size: 1.5rem;
  font-weight: 800; color: #fff;
}
.hero-stat-label {
  font-size: 0.65rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase;
}

/* ── GLOBAL BUTTONS ── */
.stButton > button {
  background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
  color: white !important; border: none !important;
  border-radius: 8px !important; font-size: 0.76rem !important;
  font-weight: 600 !important; letter-spacing: 1px !important;
  text-transform: uppercase !important; padding: 10px 22px !important;
  box-shadow: 0 4px 20px rgba(108,99,255,0.25) !important;
  transition: all 0.2s !important; width: auto !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: none !important; }

/* ── CTA ROW ── */
.cta-row { padding: 28px 40px 0; display: flex; gap: 12px; align-items: center; }
.cta-row .stButton > button {
  background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
  padding: 12px 28px !important;
  box-shadow: 0 6px 24px rgba(108,99,255,0.3) !important;
}

/* ── DIVIDER ── */
.divider { height: 1px; background: var(--border); margin: 0; }

/* ── BROWSE SECTION ── */
.browse-section { padding: 44px 40px; }
.section-eyebrow {
  font-size: 0.6rem; letter-spacing: 3.5px; text-transform: uppercase;
  color: var(--accent2); font-weight: 600; margin-bottom: 6px;
  display: flex; align-items: center; gap: 8px;
}
.section-eyebrow::before { content:''; width:16px; height:1px; background:var(--accent2); }
.section-heading {
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: 1.6rem; letter-spacing: -0.5px; color: #fff; margin-bottom: 28px;
}

/* ── MOVIE CARD ── */
.mcard {
  border-radius: 10px; overflow: hidden;
  background: var(--surface);
  border: 1px solid var(--border);
  transition: transform 0.22s cubic-bezier(.22,.68,0,1.2), box-shadow 0.22s ease, border-color 0.2s;
  position: relative; cursor: pointer;
}
.mcard:hover {
  transform: translateY(-8px) scale(1.01);
  box-shadow: 0 20px 48px rgba(0,0,0,0.7), 0 0 0 1px rgba(108,99,255,0.25);
  border-color: rgba(108,99,255,0.3);
}
.mcard-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.mcard-placeholder {
  width: 100%; aspect-ratio: 2/3;
  background: linear-gradient(160deg, #0f1020, #1a1835);
  display: flex; align-items: center; justify-content: center; font-size: 2.2rem;
}
.mcard-overlay {
  position: absolute; inset: 0;
  background: linear-gradient(to top, rgba(5,6,9,0.95) 0%, rgba(5,6,9,0.4) 40%, transparent 65%);
  opacity: 0; transition: opacity 0.22s;
}
.mcard:hover .mcard-overlay { opacity: 1; }
.mcard-play {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  width: 44px; height: 44px; border-radius: 50%;
  background: rgba(108,99,255,0.9); display: flex; align-items: center;
  justify-content: center; font-size: 1rem; opacity: 0; transition: opacity 0.22s;
  backdrop-filter: blur(4px);
}
.mcard:hover .mcard-play { opacity: 1; }
.mcard-fav {
  position: absolute; top: 8px; right: 8px;
  width: 30px; height: 30px; border-radius: 50%;
  background: rgba(5,6,9,0.6); backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.85rem; opacity: 0; transition: opacity 0.22s; z-index: 2;
}
.mcard:hover .mcard-fav { opacity: 1; }
.mcard-body { padding: 10px 11px 12px; }
.mcard-title {
  font-size: 0.75rem; font-weight: 600; color: #d0d0e0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px;
}
.mcard-genre { font-size: 0.62rem; color: var(--dim); }

/* invisible click button on each card */
.mcard-wrap { position: relative; }
.mcard-wrap .stButton { position: absolute; inset: 0; }
.mcard-wrap .stButton > button {
  position: absolute !important; inset: 0 !important;
  width: 100% !important; height: 100% !important;
  background: transparent !important; border: none !important;
  box-shadow: none !important; color: transparent !important;
  font-size: 0 !important; padding: 0 !important; cursor: pointer !important;
  border-radius: 10px !important; transform: none !important;
}
.mcard-wrap .stButton > button:hover {
  background: transparent !important; transform: none !important;
}

/* ── WATCHLIST BANNER ── */
.wl-banner {
  margin: 0 40px 40px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 20px 28px;
  display: flex; align-items: center; gap: 16px;
}
.wl-icon { font-size: 1.6rem; }
.wl-text-title { font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#fff; }
.wl-text-sub { font-size:0.75rem; color: var(--muted); margin-top:2px; }

/* ── BOTTOM BANNER ── */
.bottom-banner {
  margin: 48px 40px;
  background: linear-gradient(135deg, #0c0b22, #130824);
  border: 1px solid rgba(108,99,255,0.14);
  border-radius: 14px; padding: 40px 44px;
  display: flex; align-items: center; justify-content: space-between; gap: 32px;
  position: relative; overflow: hidden;
}
.bottom-banner::before {
  content:''; position:absolute; right:-60px; top:-60px;
  width:280px; height:280px; border-radius:50%;
  background: radial-gradient(circle, rgba(108,99,255,0.1) 0%, transparent 65%);
}
.bottom-banner-title {
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: 1.5rem; letter-spacing: -0.5px; color: #fff; margin-bottom: 8px;
}
.bottom-banner-sub { font-size: 0.82rem; color: var(--muted); font-weight: 300; }
.banner-btn .stButton > button {
  background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
  padding: 12px 28px !important; white-space: nowrap !important;
  width: auto !important; min-width: 0 !important;
}

/* ── FOR YOU PAGE ── */
.foryou-section { padding: 44px 40px; }

/* ── REC CARD ── */
.rec-card {
  background: var(--surface); border-radius: 12px;
  border: 1px solid var(--border);
  padding: 16px 18px; margin-bottom: 12px;
  display: flex; gap: 16px; align-items: flex-start;
  transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s;
}
.rec-card:hover {
  border-color: rgba(108,99,255,0.28);
  transform: translateX(4px);
  box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.rec-num {
  font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800;
  color: rgba(108,99,255,0.18); min-width: 36px; line-height: 1; text-align: right;
}
.rec-poster {
  width: 52px; height: 78px; border-radius: 7px;
  object-fit: cover; flex-shrink: 0; background: #1a1835;
}
.rec-body { flex: 1; min-width: 0; }
.rec-title { font-size: 0.9rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.rec-pills { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }
.rec-pill {
  background: rgba(108,99,255,0.1); color: var(--accent2);
  font-size: 0.6rem; padding: 2px 8px; border-radius: 4px;
  border: 1px solid rgba(108,99,255,0.18); font-weight: 500;
}
.rec-bar-bg { height: 3px; background: var(--dim); border-radius: 2px; margin-bottom: 5px; }
.rec-bar { height: 3px; background: linear-gradient(90deg, #6c63ff, #a78bfa, #f472b6); border-radius: 2px; }
.rec-score { font-size: 0.68rem; color: var(--accent2); font-weight: 500; }
.rec-view-btn .stButton > button {
  background: transparent !important; color: var(--accent2) !important;
  border: 1px solid rgba(108,99,255,0.22) !important;
  border-radius: 6px !important; font-size: 0.68rem !important;
  font-weight: 600 !important; letter-spacing: 0.5px !important;
  text-transform: none !important; padding: 5px 14px !important;
  box-shadow: none !important; width: auto !important; margin-top: 6px !important;
}
.rec-view-btn .stButton > button:hover {
  background: rgba(108,99,255,0.1) !important;
}

/* ── DETAIL PAGE ── */
.detail-backdrop { width: 100%; height: 240px; position: relative; overflow: hidden; }
.detail-backdrop img { width: 100%; height: 100%; object-fit: cover; opacity: 0.2; display: block; }
.detail-backdrop-fade {
  position: absolute; bottom: 0; left: 0; right: 0; height: 140px;
  background: linear-gradient(to top, var(--bg), transparent);
}
.detail-wrap { padding: 0 40px 48px; }
.detail-inner {
  display: flex; gap: 36px; align-items: flex-start;
  margin-top: -72px; position: relative; z-index: 2;
}
.detail-poster {
  width: 190px; min-width: 190px; border-radius: 12px;
  box-shadow: 0 20px 56px rgba(0,0,0,0.8); display: block;
}
.detail-poster-ph {
  width: 190px; min-width: 190px; height: 285px; border-radius: 12px;
  background: var(--surface2); display: flex; align-items: center;
  justify-content: center; font-size: 3rem;
}
.detail-info { flex: 1; padding-top: 76px; }
.detail-title {
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: clamp(1.8rem, 3vw, 2.8rem); letter-spacing: -0.5px;
  color: #fff; margin-bottom: 12px; line-height: 1.05;
}
.detail-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
.detail-rating {
  background: rgba(108,99,255,0.12); color: var(--accent2);
  font-size: 0.76rem; font-weight: 700; padding: 4px 12px;
  border-radius: 6px; border: 1px solid rgba(108,99,255,0.22);
}
.detail-year, .detail-runtime { font-size: 0.8rem; color: var(--muted); }
.detail-tagline { font-style: italic; color: var(--dim); font-size: 0.85rem; margin-bottom: 16px; }
.detail-overview {
  font-size: 0.88rem; color: #7878a0; line-height: 1.8;
  max-width: 640px; margin-bottom: 20px; font-weight: 300;
}
.detail-pill {
  display: inline-block; background: rgba(255,255,255,0.04);
  color: #aaa; font-size: 0.7rem; padding: 3px 12px;
  border-radius: 5px; border: 1px solid rgba(255,255,255,0.07);
  margin-right: 6px; margin-bottom: 6px;
}
.detail-sub-section {
  padding: 32px 40px; border-top: 1px solid rgba(255,255,255,0.04);
}
.detail-sub-title {
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: 1.05rem; letter-spacing: 0px; color: #fff; margin-bottom: 20px;
}
.providers { display: flex; flex-wrap: wrap; gap: 12px; }
.provider {
  background: var(--surface); border-radius: 10px;
  border: 1px solid var(--border); padding: 14px 18px;
  display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 92px;
  transition: border-color 0.15s;
}
.provider:hover { border-color: rgba(108,99,255,0.25); }
.provider img { width: 44px; height: 44px; border-radius: 8px; object-fit: cover; }
.provider-name { font-size: 0.65rem; color: #777; text-align: center; }
.provider-type { font-size: 0.58rem; color: var(--muted); }
.cast-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 10px;
}
.cast-card {
  background: var(--surface); border-radius: 8px; overflow: hidden;
  border: 1px solid var(--border); transition: border-color 0.15s;
}
.cast-card:hover { border-color: rgba(108,99,255,0.2); }
.cast-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; background: var(--surface2); }
.cast-ph { width: 100%; aspect-ratio: 2/3; background: var(--surface2); display: flex; align-items: center; justify-content: center; font-size: 1.6rem; }
.cast-name { font-size: 0.67rem; font-weight: 600; color: #bbb; padding: 6px 7px 2px; }
.cast-char { font-size: 0.59rem; color: var(--muted); padding: 0 7px 8px; }
.trailer-link {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(239,68,68,0.1); color: #f87171;
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;
  padding: 10px 20px; border-radius: 8px; border: 1px solid rgba(239,68,68,0.2);
  text-decoration: none; transition: background 0.15s;
}
.trailer-link:hover { background: rgba(239,68,68,0.15); }
.back-btn .stButton > button {
  background: transparent !important; color: var(--muted) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 7px !important; font-size: 0.76rem !important;
  font-weight: 500 !important; text-transform: none !important;
  letter-spacing: 0 !important; padding: 7px 18px !important;
  box-shadow: none !important; width: auto !important;
}

/* ── WATCHLIST PAGE ── */
.wl-empty {
  text-align: center; padding: 80px 40px; color: var(--muted);
}
.wl-empty-icon { font-size: 3rem; margin-bottom: 16px; }
.wl-empty-title { font-family:'Syne',sans-serif; font-size:1.2rem; color:#fff; margin-bottom:8px; }
.wl-empty-sub { font-size:0.85rem; }

/* ── INPUTS ── */
.stTextInput input {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 8px !important; font-size: 0.86rem !important;
}
.stTextInput input:focus {
  border-color: rgba(108,99,255,0.4) !important;
  box-shadow: 0 0 0 2px rgba(108,99,255,0.1) !important;
}
div[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 8px !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* ── CHECKBOX GENRE PILLS ── */
.stCheckbox { margin: 0 !important; }
.stCheckbox label {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 7px !important; padding: 6px 12px !important;
  font-size: 0.74rem !important; font-weight: 500 !important;
  color: var(--muted) !important; cursor: pointer !important;
  transition: border-color 0.15s, color 0.15s !important;
  white-space: nowrap !important;
}
.stCheckbox label:hover { border-color: rgba(108,99,255,0.35) !important; color: var(--text) !important; }
input[type="checkbox"]:checked + div { color: var(--accent2) !important; }

/* ── RESPONSIVE ── */
@media (max-width: 900px) {
  .nav { padding: 0 20px; }
  .hero { padding: 56px 20px 48px; }
  .browse-section, .foryou-section { padding: 32px 20px; }
  .detail-wrap { padding: 0 20px 40px; }
  .detail-inner { flex-direction: column; margin-top: -40px; }
  .detail-poster, .detail-poster-ph { width: 140px; min-width: 140px; }
  .detail-info { padding-top: 16px; }
  .bottom-banner { flex-direction: column; margin: 24px 20px; padding: 28px 24px; }
  .wl-banner { margin: 0 20px 28px; }
  div[data-testid="stHorizontalBlock"]:first-of-type { padding-left: 200px !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────
TMDB_KEY = "e27947a52a97677530fdd9e476e8b17c"
IMG_BASE  = "https://image.tmdb.org/t/p"

GENRE_ICON = {
    'Action':'💥','Adventure':'🗺️','Animation':'🎨','Comedy':'😂',
    'Crime':'🕵️','Documentary':'📽️','Drama':'🎭','Fantasy':'🧙',
    'Horror':'👻','Musical':'🎵','Mystery':'🔍','Romance':'❤️',
    'Sci-Fi':'👽','Thriller':'🔪','War':'⚔️','Western':'🤠','Children':'🧒',
}

# ── FIX: poster fetch with robust fallback ──
@st.cache_data(show_spinner=False)
def tmdb_search(title):
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_KEY, "query": clean},
            timeout=5
        )
        if r.status_code != 200:
            return None
        results = r.json().get('results', [])
        # prefer results that have a poster
        with_poster = [x for x in results if x.get('poster_path')]
        return with_poster[0] if with_poster else (results[0] if results else None)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def tmdb_details(title):
    result = tmdb_search(title)
    if not result:
        return None
    mid = result['id']
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{mid}",
            params={"api_key": TMDB_KEY,
                    "append_to_response": "credits,watch/providers,videos"},
            timeout=6
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def poster_url(title):
    r = tmdb_search(title)
    if r and r.get('poster_path'):
        return f"{IMG_BASE}/w300{r['poster_path']}"
    return None

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    rp = os.path.join("data", "ratings.csv")
    mp = os.path.join("data", "movies.csv")
    if not os.path.exists(rp) or not os.path.exists(mp):
        return None, None
    rd = pd.read_csv(rp)
    md = pd.read_csv(mp)
    md['genres'] = md['genres'].fillna('')
    return rd, md

# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE  (improved scoring)
# ─────────────────────────────────────────────
def genre_recs(picked, movies_df, n=12):
    """
    Score = TF-IDF cosine similarity weighted by genre coverage.
    This spreads scores apart more meaningfully than raw cosine alone.
    """
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf.fit(movies_df['genres'])
    q = tfidf.transform(['|'.join(picked)])
    m = tfidf.transform(movies_df['genres'])
    cosine = cosine_similarity(q, m).flatten()

    # bonus: fraction of picked genres that appear in movie
    def coverage(movie_genres):
        mg = set(g.strip() for g in movie_genres.split('|'))
        return len(set(picked) & mg) / len(picked)

    cov = movies_df['genres'].apply(coverage).values
    # weighted blend — cosine 60%, genre coverage 40%
    combined = 0.6 * cosine + 0.4 * cov

    # normalise to 0–1
    mx = combined.max()
    if mx > 0:
        combined = combined / mx

    idx = combined.argsort()[::-1][:n]
    return pd.DataFrame([{
        'title':  movies_df.iloc[i]['title'],
        'genres': movies_df.iloc[i]['genres'],
        'score':  round(float(combined[i]), 4)
    } for i in idx])


def search_recs(query, movies_df, n=12):
    """Text-based search across title + genres using TF-IDF."""
    movies_df = movies_df.copy()
    movies_df['search_text'] = (
        movies_df['title'].fillna('') + ' ' +
        movies_df['genres'].fillna('').str.replace('|', ' ')
    )
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    mat   = tfidf.fit_transform(movies_df['search_text'])
    q_vec = tfidf.transform([query])
    scores = cosine_similarity(q_vec, mat).flatten()

    # normalise
    mx = scores.max()
    if mx > 0:
        scores = scores / mx

    idx = scores.argsort()[::-1][:n]
    return pd.DataFrame([{
        'title':  movies_df.iloc[i]['title'],
        'genres': movies_df.iloc[i]['genres'],
        'score':  round(float(scores[i]), 4)
    } for i in idx if scores[i] > 0])

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    'page':      'home',
    'prev':      'home',
    'movie':     None,
    'genres':    [],
    'recs':      None,
    'watchlist': [],          # NEW
    'rec_mode':  'genre',     # NEW: 'genre' | 'search'
    'search_q':  '',          # NEW
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ratings_df, movies_df = load_data()

# ─────────────────────────────────────────────
# NAV BAR (fixed overlap bug — renders before columns)
# ─────────────────────────────────────────────
st.markdown("""
<div class="nav">
    <div class="nav-logo">MovieRecSys</div>
    <div class="nav-divider"></div>
</div>
""", unsafe_allow_html=True)

# Nav buttons float into the sticky nav via CSS margin-top trick
wl_count   = len(st.session_state.watchlist)
wl_label   = f"Watchlist ({wl_count})" if wl_count else "Watchlist"
c1, c2, c3, _sp = st.columns([1, 1, 1.2, 14])
with c1:
    if st.button("Browse", key="nav_browse"):
        st.session_state.page  = 'home'
        st.session_state.movie = None
        st.rerun()
with c2:
    if st.button("For You", key="nav_foryou"):
        st.session_state.page  = 'recs'
        st.session_state.movie = None
        st.rerun()
with c3:
    if st.button(wl_label, key="nav_watchlist"):
        st.session_state.page  = 'watchlist'
        st.session_state.movie = None
        st.rerun()

if ratings_df is None:
    st.error("Data files not found. Add data/ratings.csv and data/movies.csv")
    st.stop()

all_genres = sorted({
    g.strip() for gs in movies_df['genres']
    for g in gs.split('|')
    if g.strip() not in ('', '(no genres listed)')
})

# ─────────────────────────────────────────────
# DETAIL PAGE
# ─────────────────────────────────────────────
def show_detail(title):
    st.markdown('<div style="padding:20px 40px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Back", key="back_btn"):
        st.session_state.movie = None
        st.session_state.page  = st.session_state.prev
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

    det = tmdb_details(title)
    if not det or 'title' not in det:
        st.warning("Could not load movie details from TMDB.")
        return

    ptitle   = det.get('title', title)
    overview = det.get('overview', '')
    tagline  = det.get('tagline', '')
    year     = det.get('release_date', '')[:4]
    rating   = round(det.get('vote_average', 0), 1)
    rt       = det.get('runtime') or 0
    runtime  = f"{rt//60}h {rt%60}m" if rt else ''
    genres   = [g['name'] for g in det.get('genres', [])]
    poster   = f"{IMG_BASE}/w400{det['poster_path']}"    if det.get('poster_path')   else None
    backdrop = f"{IMG_BASE}/w1280{det['backdrop_path']}" if det.get('backdrop_path') else None

    if backdrop:
        st.markdown(f"""
        <div class="detail-backdrop">
            <img src="{backdrop}" alt="backdrop"/>
            <div class="detail-backdrop-fade"></div>
        </div>""", unsafe_allow_html=True)

    pimg  = f'<img class="detail-poster" src="{poster}" alt="{ptitle}"/>' if poster \
            else '<div class="detail-poster-ph">🎬</div>'
    gpills = ''.join(f'<span class="detail-pill">{g}</span>' for g in genres)

    st.markdown(f"""
    <div class="detail-wrap">
        <div class="detail-inner">
            {pimg}
            <div class="detail-info">
                <div class="detail-title">{ptitle}</div>
                <div class="detail-row">
                    <span class="detail-year">{year}</span>
                    <span class="detail-rating">★ {rating} / 10</span>
                    <span class="detail-runtime">{runtime}</span>
                </div>
                {'<div class="detail-tagline">"' + tagline + '"</div>' if tagline else ''}
                <div class="detail-overview">{overview}</div>
                <div>{gpills}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Watchlist button
    in_wl = title in st.session_state.watchlist
    wl_lbl = "✓ In Watchlist" if in_wl else "+ Add to Watchlist"
    st.markdown('<div style="padding:0 40px 16px;">', unsafe_allow_html=True)
    if st.button(wl_lbl, key=f"wl_detail_{title}"):
        if in_wl:
            st.session_state.watchlist.remove(title)
        else:
            st.session_state.watchlist.append(title)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Trailer
    videos  = det.get('videos', {}).get('results', [])
    trailer = next((v for v in videos if v.get('type') == 'Trailer'
                    and v.get('site') == 'YouTube'), None)
    if trailer:
        st.markdown(f"""
        <div class="detail-sub-section" style="border-top:none;padding-top:0;padding-bottom:24px;">
            <a class="trailer-link"
               href="https://www.youtube.com/watch?v={trailer['key']}"
               target="_blank">▶ &nbsp; Watch Trailer on YouTube</a>
        </div>""", unsafe_allow_html=True)

    # Where to Watch (India-first)
    providers_data = det.get('watch/providers', {}).get('results', {})
    region = providers_data.get('IN', providers_data.get('US', {}))
    seen, combined = set(), []
    for label, lst in [('Stream', region.get('flatrate', [])),
                       ('Rent',   region.get('rent',     [])),
                       ('Buy',    region.get('buy',      []))]:
        for p in lst:
            if p['provider_name'] not in seen:
                seen.add(p['provider_name'])
                combined.append((p, label))

    st.markdown('<div class="detail-sub-section">', unsafe_allow_html=True)
    st.markdown('<div class="detail-sub-title">Where to Watch</div>', unsafe_allow_html=True)
    if combined:
        cards = ''
        for p, label in combined[:10]:
            logo = f'{IMG_BASE}/w92{p["logo_path"]}' if p.get('logo_path') else None
            img  = f'<img src="{logo}" alt="{p["provider_name"]}"/>' if logo \
                   else '<div style="width:44px;height:44px;background:#1a1835;border-radius:8px;"></div>'
            cards += f"""<div class="provider">
                {img}
                <div class="provider-name">{p['provider_name']}</div>
                <div class="provider-type">{label}</div>
            </div>"""
        st.markdown(f'<div class="providers">{cards}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--muted);font-size:0.82rem;">No streaming data available for your region.</p>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Cast
    cast = det.get('credits', {}).get('cast', [])[:14]
    if cast:
        cast_html = ''
        for c in cast:
            img = f'{IMG_BASE}/w185{c["profile_path"]}' if c.get('profile_path') else None
            ph  = f'<img class="cast-img" src="{img}" alt="{c["name"]}"/>' if img \
                  else '<div class="cast-ph">👤</div>'
            cast_html += f"""<div class="cast-card">
                {ph}
                <div class="cast-name">{c['name']}</div>
                <div class="cast-char">{c.get('character','')[:22]}</div>
            </div>"""
        st.markdown(f"""
        <div class="detail-sub-section">
            <div class="detail-sub-title">Cast</div>
            <div class="cast-grid">{cast_html}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if st.session_state.movie:
    show_detail(st.session_state.movie)
    st.stop()

# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if st.session_state.page == 'home':

    total_movies = len(movies_df)
    total_ratings = len(ratings_df)

    st.markdown(f"""
    <div class="hero">
        <div class="hero-glow-1"></div>
        <div class="hero-glow-2"></div>
        <div class="hero-label">AI-Powered Discovery</div>
        <div class="hero-title">Discover <em>Cinema</em><br>You Will Love</div>
        <div class="hero-sub">
            Browse thousands of movies or let our recommendation engine surface
            films tailored to your taste — no account needed.
        </div>
        <div class="hero-stats">
            <div>
                <div class="hero-stat-num">{total_movies:,}</div>
                <div class="hero-stat-label">Movies</div>
            </div>
            <div>
                <div class="hero-stat-num">{total_ratings:,}</div>
                <div class="hero-stat-label">Ratings</div>
            </div>
            <div>
                <div class="hero-stat-num">{len(all_genres)}</div>
                <div class="hero-stat-label">Genres</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cta-row">', unsafe_allow_html=True)
    if st.button("Get Personalised Recommendations →", key="hero_cta"):
        st.session_state.page = 'recs'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Watchlist mini-banner
    if st.session_state.watchlist:
        wl_titles = ', '.join(st.session_state.watchlist[:3])
        more = f" +{len(st.session_state.watchlist)-3} more" if len(st.session_state.watchlist) > 3 else ""
        st.markdown(f"""
        <div class="wl-banner">
            <div class="wl-icon">🎯</div>
            <div>
                <div class="wl-text-title">Your Watchlist ({len(st.session_state.watchlist)})</div>
                <div class="wl-text-sub">{wl_titles}{more}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Browse
    st.markdown('<div class="browse-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Explore</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Browse Movies</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        search = st.text_input("", placeholder="Search by title…",
                               label_visibility="collapsed", key="search")
    with fc2:
        gpick = st.selectbox("", ['All'] + all_genres,
                             label_visibility="collapsed", key="gpick")

    if search:
        pool = movies_df[movies_df['title'].str.contains(search, case=False, na=False)]
    elif gpick != 'All':
        pool = movies_df[movies_df['genres'].str.contains(gpick, case=False, na=False)]
    else:
        pool = movies_df.sample(min(40, len(movies_df)), random_state=42)
    filtered = pool.head(40)

    st.markdown(
        f'<div style="font-size:0.66rem;color:var(--muted);letter-spacing:2px;margin-bottom:22px;font-weight:600;">'
        f'{len(filtered)} TITLES</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(8)
    for i, (_, row) in enumerate(filtered.iterrows()):
        title  = row['title']
        short  = title[:16] + '…' if len(title) > 16 else title
        genre1 = row['genres'].split('|')[0].strip() if row['genres'] else ''
        purl   = poster_url(title)
        icon   = GENRE_ICON.get(genre1, '🎬')
        in_wl  = title in st.session_state.watchlist

        with cols[i % 8]:
            img_html = (
                f'<img class="mcard-img" src="{purl}" loading="lazy" alt="{short}" '
                f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                f'<div class="mcard-placeholder" style="display:none">{icon}</div>'
                if purl else
                f'<div class="mcard-placeholder">{icon}</div>'
            )
            fav_style = "opacity:1" if in_wl else ""
            st.markdown(f"""
            <div class="mcard">
                {img_html}
                <div class="mcard-overlay"></div>
                <div class="mcard-play">▶</div>
                <div class="mcard-fav" style="{fav_style}">{"❤️" if in_wl else "🤍"}</div>
                <div class="mcard-body">
                    <div class="mcard-title">{short}</div>
                    <div class="mcard-genre">{genre1}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button("", key=f"card_{i}", help=title):
                st.session_state.movie = title
                st.session_state.prev  = 'home'
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="bottom-banner">
        <div class="bottom-banner-text">
            <div class="bottom-banner-title">Not sure what to watch?</div>
            <div class="bottom-banner-sub">
                Tell us your favourite genres — or describe a movie — and we will
                find the perfect films for you.
            </div>
        </div>
        <div class="banner-btn">""", unsafe_allow_html=True)
    if st.button("Find My Movies →", key="banner_cta"):
        st.session_state.page = 'recs'
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WATCHLIST PAGE  (new)
# ─────────────────────────────────────────────
elif st.session_state.page == 'watchlist':
    st.markdown("""
    <div class="hero" style="padding-bottom:52px;">
        <div class="hero-glow-1"></div>
        <div class="hero-label">Your Collection</div>
        <div class="hero-title">My <em>Watchlist</em></div>
        <div class="hero-sub">Movies you've saved to watch later.</div>
    </div>""", unsafe_allow_html=True)

    wl = st.session_state.watchlist
    if not wl:
        st.markdown("""
        <div class="wl-empty">
            <div class="wl-empty-icon">🎬</div>
            <div class="wl-empty-title">Your watchlist is empty</div>
            <div class="wl-empty-sub">Browse movies and hit "Add to Watchlist" to save them here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="browse-section">', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.66rem;color:var(--muted);letter-spacing:2px;margin-bottom:24px;font-weight:600;">'
            f'{len(wl)} SAVED</div>', unsafe_allow_html=True
        )
        cols = st.columns(8)
        for i, title in enumerate(wl):
            genre_row = movies_df[movies_df['title'] == title]
            genre1 = genre_row['genres'].values[0].split('|')[0].strip() if len(genre_row) else ''
            purl  = poster_url(title)
            icon  = GENRE_ICON.get(genre1, '🎬')
            short = title[:16] + '…' if len(title) > 16 else title

            with cols[i % 8]:
                img_html = (
                    f'<img class="mcard-img" src="{purl}" loading="lazy" alt="{short}" '
                    f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                    f'<div class="mcard-placeholder" style="display:none">{icon}</div>'
                    if purl else f'<div class="mcard-placeholder">{icon}</div>'
                )
                st.markdown(f"""
                <div class="mcard">
                    {img_html}
                    <div class="mcard-overlay"></div>
                    <div class="mcard-play">▶</div>
                    <div class="mcard-body">
                        <div class="mcard-title">{short}</div>
                        <div class="mcard-genre">{genre1}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
                if st.button("", key=f"wlcard_{i}", help=title):
                    st.session_state.movie = title
                    st.session_state.prev  = 'watchlist'
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="padding:0 40px 48px;">', unsafe_allow_html=True)
        if st.button("Clear Watchlist", key="clear_wl"):
            st.session_state.watchlist = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOR YOU PAGE
# ─────────────────────────────────────────────
elif st.session_state.page == 'recs':

    st.markdown("""
    <div class="hero" style="padding-bottom:52px;">
        <div class="hero-glow-1"></div>
        <div class="hero-glow-2"></div>
        <div class="hero-label">Personalised</div>
        <div class="hero-title">Made <em>For You</em></div>
        <div class="hero-sub">
            Pick genres you love, or describe what you're in the mood for —
            we'll find the perfect match.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="foryou-section">', unsafe_allow_html=True)

    # ── Mode toggle ──────────────────────────
    st.markdown('<div class="section-eyebrow" style="margin-bottom:14px;">Recommendation Mode</div>',
                unsafe_allow_html=True)
    mode_c1, mode_c2, _ = st.columns([1, 1, 6])
    with mode_c1:
        if st.button("🎭 By Genre", key="mode_genre"):
            st.session_state.rec_mode = 'genre'
            st.session_state.recs     = None
    with mode_c2:
        if st.button("🔍 By Search", key="mode_search"):
            st.session_state.rec_mode = 'search'
            st.session_state.recs     = None

    mode_active = st.session_state.rec_mode
    st.markdown(
        f'<div style="font-size:0.7rem;color:var(--accent2);margin:8px 0 28px;letter-spacing:1px;font-weight:600;">'
        f'ACTIVE MODE: {"Genre-based" if mode_active=="genre" else "Search-based"}</div>',
        unsafe_allow_html=True
    )

    # ── GENRE MODE ───────────────────────────
    if mode_active == 'genre':
        st.markdown(
            '<div class="section-eyebrow" style="margin-bottom:14px;">Step 1 — Pick genres you love</div>',
            unsafe_allow_html=True
        )

        # FIX: use stable checkbox keys tied to genre name and re-read from state each render
        gcols  = st.columns(9)
        picked = []
        for idx, g in enumerate(all_genres):
            icon    = GENRE_ICON.get(g, '🎬')
            cb_key  = f"gc_{g}"
            default = g in st.session_state.genres
            with gcols[idx % 9]:
                checked = st.checkbox(f"{icon} {g}", key=cb_key, value=default)
                if checked:
                    picked.append(g)

        # persist immediately so selections survive reruns
        st.session_state.genres = picked

        if picked:
            st.markdown(
                f'<div style="font-size:0.7rem;color:var(--accent2);margin:14px 0 4px;'
                f'letter-spacing:1px;font-weight:600;">SELECTED: {" · ".join(picked)}</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-eyebrow" style="margin-top:28px;margin-bottom:10px;">Step 2 — How many results?</div>',
            unsafe_allow_html=True
        )
        top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn")

        b1, b2, _ = st.columns([1, 1.2, 8])
        with b1:
            find_btn = st.button("Find My Movies", key="find_btn")
        with b2:
            if st.button("Back to Browse", key="back_browse"):
                st.session_state.page = 'home'
                st.rerun()

        if find_btn:
            if not picked:
                st.warning("Please select at least one genre.")
            else:
                with st.spinner("Finding movies…"):
                    st.session_state.recs = genre_recs(picked, movies_df, top_n)

    # ── SEARCH MODE ──────────────────────────
    else:
        st.markdown(
            '<div class="section-eyebrow" style="margin-bottom:14px;">Describe what you want to watch</div>',
            unsafe_allow_html=True
        )
        sq = st.text_input(
            "", placeholder="e.g. 'space adventure with humour' or 'dark crime thriller'",
            label_visibility="collapsed", key="search_query_input",
            value=st.session_state.search_q
        )
        st.session_state.search_q = sq

        top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn_s")

        b1, b2, _ = st.columns([1, 1.2, 8])
        with b1:
            srch_btn = st.button("Search Movies", key="srch_btn")
        with b2:
            if st.button("Back to Browse", key="back_browse_s"):
                st.session_state.page = 'home'
                st.rerun()

        if srch_btn:
            if not sq.strip():
                st.warning("Please enter a search description.")
            else:
                with st.spinner("Searching…"):
                    st.session_state.recs = search_recs(sq, movies_df, top_n)

    # ── RESULTS ──────────────────────────────
    if st.session_state.recs is not None:
        recs = st.session_state.recs
        if recs.empty:
            st.warning("No results found. Try different genres or a different search query.")
        else:
            st.markdown('<div class="divider" style="margin:36px 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-eyebrow">Your Results</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-heading">Top {len(recs)} Picks For You</div>',
                        unsafe_allow_html=True)

            lc, rc = st.columns([1.1, 1])

            with lc:
                for i, row in recs.iterrows():
                    bw    = int(row['score'] * 100)
                    pct   = f"{bw}%"
                    pills = ''.join(
                        f'<span class="rec-pill">{g.strip()}</span>'
                        for g in row['genres'].split('|') if g.strip()
                    )
                    pimg  = poster_url(row['title'])
                    g0    = row['genres'].split('|')[0].strip()
                    ph    = (
                        f'<img class="rec-poster" src="{pimg}" alt="poster" '
                        f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                        f'<div class="rec-poster" style="display:none;align-items:center;'
                        f'justify-content:center;font-size:1.4rem;">{GENRE_ICON.get(g0,"🎬")}</div>'
                        if pimg else
                        f'<div class="rec-poster" style="display:flex;align-items:center;'
                        f'justify-content:center;font-size:1.4rem;">{GENRE_ICON.get(g0,"🎬")}</div>'
                    )

                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-num">{'0' if i+1 < 10 else ''}{i+1}</div>
                        {ph}
                        <div class="rec-body">
                            <div class="rec-title">{row['title']}</div>
                            <div class="rec-pills">{pills}</div>
                            <div class="rec-bar-bg">
                                <div class="rec-bar" style="width:{pct}"></div>
                            </div>
                            <div class="rec-score">Match Score: {row['score']:.2f}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    rcol1, rcol2, _ = st.columns([1, 1, 4])
                    with rcol1:
                        st.markdown('<div class="rec-view-btn">', unsafe_allow_html=True)
                        if st.button("View Details", key=f"rv_{i}"):
                            st.session_state.movie = row['title']
                            st.session_state.prev  = 'recs'
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                    with rcol2:
                        in_wl  = row['title'] in st.session_state.watchlist
                        wl_lbl = "✓ Saved" if in_wl else "+ Watchlist"
                        st.markdown('<div class="rec-view-btn">', unsafe_allow_html=True)
                        if st.button(wl_lbl, key=f"wl_{i}"):
                            if in_wl:
                                st.session_state.watchlist.remove(row['title'])
                            else:
                                st.session_state.watchlist.append(row['title'])
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

            with rc:
                fig, ax = plt.subplots(figsize=(6, max(4, len(recs) * 0.55)))
                fig.patch.set_facecolor('#050609')
                ax.set_facecolor('#0c0d14')
                titles = [r['title'][:24] + '…' if len(r['title']) > 24
                          else r['title'] for _, r in recs.iterrows()]
                scores = recs['score'].values
                palette = ['#6c63ff','#7c75ff','#8f87ff','#a78bfa','#b89bfb',
                           '#c9acfc','#d4b4fd','#e0bdfe','#f472b6','#f896c8',
                           '#6c63ff','#7c75ff','#8f87ff','#a78bfa','#b89bfb',
                           '#c9acfc','#d4b4fd','#e0bdfe','#f472b6','#f896c8']
                bars = ax.barh(titles[::-1], scores[::-1],
                               color=palette[:len(scores)][::-1], height=0.56,
                               edgecolor='none')
                for b in bars:
                    ax.text(b.get_width() + 0.008, b.get_y() + b.get_height() / 2,
                            f'{b.get_width():.2f}', va='center',
                            color='#55556a', fontsize=7.5)
                ax.set_xlim(0, 1.15)
                ax.set_xlabel('Match Score', color='#44445a', fontsize=8)
                ax.set_title('Match Chart', color='#aaa', fontsize=10,
                             fontweight='bold', pad=12, fontfamily='sans-serif')
                ax.tick_params(colors='#44445a', labelsize=7.5)
                for s in ax.spines.values():
                    s.set_edgecolor('#1a1a2a')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown('<br>', unsafe_allow_html=True)
                st.download_button(
                    "⬇ Download CSV",
                    recs.to_csv(index=False),
                    "recommendations.csv",
                    "text/csv",
                    key="dl_csv"
                )

    st.markdown('</div>', unsafe_allow_html=True) 
