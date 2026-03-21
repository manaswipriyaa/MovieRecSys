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
# STYLES
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
  --text:      #eaeaf5;
  --muted:     #5a5a72;
  --subtle:    #2a2a3e;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; -webkit-font-smoothing: antialiased; }
.stApp { background: var(--bg) !important; color: var(--text); }

/* Remove all default Streamlit padding */
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stSidebar"], footer, header { display: none !important; }
.stVerticalBlock { gap: 0 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--subtle); border-radius: 3px; }

/* ═══════════════════════════════════════════════
   CONTENT WRAPPER — centred with max-width
   ═══════════════════════════════════════════════ */
.page-wrap {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 48px;
}

/* ═══════════════════════════════════════════════
   NAV — Streamlit columns rendered inside a
   styled container. No CSS position tricks.
   ═══════════════════════════════════════════════ */
.nav-bar {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: rgba(8,9,14,0.97);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  padding: 0 48px;
}
.nav-inner {
  max-width: 1280px;
  margin: 0 auto;
  height: 60px;
  display: flex;
  align-items: center;
  gap: 0;
}
.nav-logo {
  font-family: 'Playfair Display', serif;
  font-weight: 900;
  font-size: 1.1rem;
  letter-spacing: 5px;
  text-transform: uppercase;
  color: var(--accent);
  margin-right: 40px;
  white-space: nowrap;
  flex-shrink: 0;
}
.nav-sep {
  width: 1px; height: 20px;
  background: var(--border);
  margin-right: 12px;
  flex-shrink: 0;
}

/* Nav buttons rendered via Streamlit — override styles */
.nav-btn-wrap .stButton > button {
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
  box-shadow: none !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.7rem !important;
  font-weight: 500 !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  padding: 8px 18px !important;
  border-radius: 5px !important;
  width: auto !important;
  min-width: 0 !important;
  white-space: nowrap !important;
  transition: color 0.15s, background 0.15s !important;
  transform: none !important;
}
.nav-btn-wrap .stButton > button:hover {
  color: var(--accent) !important;
  background: rgba(201,169,110,0.07) !important;
  transform: none !important;
}
.nav-btn-active .stButton > button {
  color: var(--accent2) !important;
}

/* ═══════════════════════════════════════════════
   HERO
   ═══════════════════════════════════════════════ */
.hero-band {
  background: linear-gradient(155deg, #0c0b1c 0%, #0f0c1f 55%, #080912 100%);
  border-bottom: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}
.hero-band::before {
  content: '';
  position: absolute; top: -100px; right: 0;
  width: 400px; height: 400px; border-radius: 50%;
  background: radial-gradient(circle, rgba(201,169,110,0.08) 0%, transparent 65%);
  pointer-events: none;
}
.hero-inner {
  max-width: 1280px;
  margin: 0 auto;
  padding: 52px 48px 44px;
}
.hero-eyebrow {
  font-size: 0.58rem; letter-spacing: 4px; text-transform: uppercase;
  color: var(--accent); font-weight: 500; margin-bottom: 14px;
}
.hero-title {
  font-family: 'Playfair Display', serif; font-weight: 900;
  font-size: clamp(2rem, 3.5vw, 3rem); color: #fff;
  line-height: 1.1; margin-bottom: 14px; letter-spacing: -0.5px;
}
.hero-title em { font-style: italic; color: var(--accent); }
.hero-sub {
  font-size: 0.88rem; color: var(--muted); line-height: 1.75;
  max-width: 440px; font-weight: 300;
}

/* ═══════════════════════════════════════════════
   GLOBAL BUTTON STYLES
   ═══════════════════════════════════════════════ */

/* Primary — gold */
.stButton > button {
  background: var(--accent) !important;
  color: #08090e !important;
  border: none !important;
  border-radius: 6px !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.73rem !important;
  font-weight: 600 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  padding: 10px 24px !important;
  box-shadow: 0 4px 16px rgba(201,169,110,0.2) !important;
  width: auto !important;
  min-width: 0 !important;
  white-space: nowrap !important;
  transition: opacity 0.15s !important;
  transform: none !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: none !important; }

/* Ghost variant */
.btn-ghost .stButton > button {
  background: transparent !important;
  color: var(--muted) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  box-shadow: none !important;
  letter-spacing: 1px !important;
  text-transform: none !important;
  font-size: 0.73rem !important;
  padding: 9px 20px !important;
}
.btn-ghost .stButton > button:hover {
  color: var(--text) !important;
  border-color: rgba(255,255,255,0.25) !important;
}

/* Watchlist outline */
.btn-outline .stButton > button {
  background: transparent !important;
  color: var(--accent2) !important;
  border: 1px solid rgba(201,169,110,0.28) !important;
  box-shadow: none !important;
  font-size: 0.72rem !important;
  padding: 9px 22px !important;
}
.btn-outline .stButton > button:hover {
  background: var(--accentdim) !important;
  border-color: rgba(201,169,110,0.55) !important;
}

/* Danger / remove */
.btn-danger .stButton > button {
  background: transparent !important;
  color: #e05c5c !important;
  border: 1px solid rgba(224,92,92,0.25) !important;
  box-shadow: none !important;
  font-size: 0.72rem !important;
  padding: 9px 20px !important;
  text-transform: none !important;
  letter-spacing: 0.5px !important;
}

/* Inline text link — rec detail */
.btn-text .stButton > button {
  background: transparent !important;
  color: var(--accent) !important;
  border: none !important;
  box-shadow: none !important;
  font-size: 0.7rem !important;
  font-weight: 500 !important;
  padding: 3px 0 !important;
  letter-spacing: 0.3px !important;
  text-transform: none !important;
  text-decoration: underline !important;
  text-underline-offset: 3px !important;
  text-decoration-color: rgba(201,169,110,0.4) !important;
}
.btn-text .stButton > button:hover {
  text-decoration-color: var(--accent) !important;
  opacity: 1 !important;
}

/* Download */
.stDownloadButton > button {
  background: transparent !important;
  color: var(--muted) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  box-shadow: none !important;
  font-size: 0.7rem !important;
  padding: 8px 18px !important;
}
.stDownloadButton > button:hover { color: var(--text) !important; }

/* ═══════════════════════════════════════════════
   MOVIE CARD
   Card is clickable via an invisible absolute button.
   NO visible ▶ button or any text below the card.
   ═══════════════════════════════════════════════ */
.card-col {
  position: relative;
  margin-bottom: 6px;
}

/* The invisible Streamlit button covers the whole card */
.card-col .stButton {
  position: absolute !important;
  inset: 0 !important;
  z-index: 10 !important;
}
.card-col .stButton > button {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  font-size: 0 !important;
  color: transparent !important;
  cursor: pointer !important;
  border-radius: 8px !important;
  transform: none !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
}
.card-col .stButton > button:hover {
  background: transparent !important;
  opacity: 1 !important;
  transform: none !important;
}

.mcard {
  border-radius: 8px;
  overflow: hidden;
  background: var(--surface);
  border: 1px solid var(--border);
  transition: transform 0.22s cubic-bezier(.22,.68,0,1.2),
              box-shadow 0.22s ease,
              border-color 0.2s;
  cursor: pointer;
  pointer-events: none; /* handled by button */
  position: relative;
}
.card-col:hover .mcard {
  transform: translateY(-7px) scale(1.012);
  box-shadow: 0 20px 50px rgba(0,0,0,0.8), 0 0 0 1px rgba(201,169,110,0.22);
  border-color: rgba(201,169,110,0.3);
}
.mcard-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.mcard-ph {
  width: 100%; aspect-ratio: 2/3;
  background: linear-gradient(160deg, var(--surface), var(--surface2));
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem; color: var(--muted);
}
.mcard-overlay {
  position: absolute; inset: 0;
  background: linear-gradient(to top,
    rgba(8,9,14,0.97) 0%,
    rgba(8,9,14,0.4) 35%,
    transparent 65%);
  opacity: 0;
  transition: opacity 0.22s;
  display: flex; flex-direction: column;
  justify-content: flex-end;
  padding: 12px 10px 11px;
  pointer-events: none;
}
.card-col:hover .mcard-overlay { opacity: 1; }
.mcard-ov-title {
  font-family: 'Playfair Display', serif;
  font-size: 0.74rem; font-weight: 700; color: #fff;
  line-height: 1.2; margin-bottom: 2px;
}
.mcard-ov-sub { font-size: 0.59rem; color: var(--accent); }
.mcard-body { padding: 9px 10px 10px; }
.mcard-title {
  font-size: 0.71rem; font-weight: 500; color: #c0c0d8;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  margin-bottom: 2px;
}
.mcard-genre { font-size: 0.59rem; color: var(--muted); }

/* ═══════════════════════════════════════════════
   SECTION HELPERS
   ═══════════════════════════════════════════════ */
.divider { height: 1px; background: var(--border); }
.section-eyebrow {
  font-size: 0.58rem; letter-spacing: 3.5px; text-transform: uppercase;
  color: var(--muted); font-weight: 500; margin-bottom: 4px;
}
.section-heading {
  font-family: 'Playfair Display', serif; font-weight: 700;
  font-size: 1.25rem; color: #fff;
  margin-bottom: 22px; letter-spacing: -0.3px;
}
.count-line {
  font-size: 0.6rem; color: var(--muted); letter-spacing: 2px;
  font-weight: 500; margin: 12px 0 18px;
}

/* ═══════════════════════════════════════════════
   BOTTOM BANNER
   ═══════════════════════════════════════════════ */
.btm-band {
  background: linear-gradient(120deg, #100e24, #17102a);
  border-top: 1px solid rgba(201,169,110,0.1);
  margin-top: 48px;
}
.btm-inner {
  max-width: 1280px; margin: 0 auto;
  padding: 36px 48px;
  display: flex; align-items: center;
  justify-content: space-between; gap: 32px;
  flex-wrap: wrap;
}
.btm-title {
  font-family: 'Playfair Display', serif; font-weight: 700;
  font-size: 1.3rem; color: #fff; margin-bottom: 6px;
}
.btm-sub { font-size: 0.82rem; color: var(--muted); font-weight: 300; }

/* ═══════════════════════════════════════════════
   DETAIL PAGE
   ═══════════════════════════════════════════════ */
.detail-backdrop { width: 100%; height: 210px; position: relative; overflow: hidden; }
.detail-backdrop img { width: 100%; height: 100%; object-fit: cover; opacity: 0.18; display: block; }
.detail-fade { position: absolute; inset: 0; background: linear-gradient(to top, var(--bg) 0%, transparent 55%); }

.detail-inner {
  max-width: 1280px; margin: 0 auto;
  padding: 0 48px 56px;
}
.detail-flex {
  display: flex; gap: 40px; align-items: flex-start;
  margin-top: -72px; position: relative; z-index: 2;
}
.detail-poster {
  width: 175px; min-width: 175px; border-radius: 10px;
  box-shadow: 0 24px 60px rgba(0,0,0,0.88); display: block;
}
.detail-poster-ph {
  width: 175px; min-width: 175px; height: 262px;
  border-radius: 10px; background: var(--surface2);
  display: flex; align-items: center; justify-content: center; font-size: 2.8rem;
}
.detail-info { flex: 1; padding-top: 80px; }
.detail-title {
  font-family: 'Playfair Display', serif; font-weight: 900;
  font-size: clamp(1.5rem, 3vw, 2.5rem); color: #fff;
  line-height: 1.08; margin-bottom: 12px;
}
.detail-meta { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
.detail-rating {
  background: rgba(201,169,110,0.12); color: var(--accent2);
  font-size: 0.73rem; font-weight: 600; padding: 3px 12px;
  border-radius: 5px; border: 1px solid rgba(201,169,110,0.22);
}
.detail-year, .detail-runtime { font-size: 0.77rem; color: var(--muted); }
.detail-tagline { font-style: italic; color: var(--muted); font-size: 0.83rem; margin-bottom: 14px; }
.detail-overview { font-size: 0.87rem; color: #7878a0; line-height: 1.8; max-width: 620px; margin-bottom: 20px; font-weight: 300; }
.detail-pill { display: inline-block; background: rgba(255,255,255,0.04); color: #aaa; font-size: 0.67rem; padding: 3px 11px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.07); margin-right: 5px; margin-bottom: 5px; }

.trailer-btn { display: inline-flex; align-items: center; gap: 8px; background: rgba(224,92,92,0.1); color: #ef8080; font-size: 0.73rem; font-weight: 600; padding: 9px 20px; border-radius: 6px; border: 1px solid rgba(224,92,92,0.2); text-decoration: none; transition: background 0.15s; }
.trailer-btn:hover { background: rgba(224,92,92,0.18); }

.sub-section { max-width: 1280px; margin: 0 auto; padding: 26px 48px; border-top: 1px solid var(--border); }
.sub-title { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 0.95rem; color: #fff; margin-bottom: 16px; }

.providers { display: flex; flex-wrap: wrap; gap: 10px; }
.provider { background: var(--surface); border-radius: 9px; border: 1px solid var(--border); padding: 12px 14px; display: flex; flex-direction: column; align-items: center; gap: 5px; min-width: 84px; text-decoration: none; transition: border-color 0.15s, transform 0.15s; }
.provider:hover { border-color: rgba(201,169,110,0.32); transform: translateY(-2px); }
.provider img { width: 40px; height: 40px; border-radius: 7px; object-fit: cover; }
.provider-name { font-size: 0.61rem; color: #888; text-align: center; }
.provider-type { font-size: 0.55rem; color: var(--muted); }

.cast-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); gap: 9px; }
.cast-card { background: var(--surface); border-radius: 7px; overflow: hidden; border: 1px solid var(--border); transition: border-color 0.15s; }
.cast-card:hover { border-color: rgba(201,169,110,0.22); }
.cast-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.cast-ph { width: 100%; aspect-ratio: 2/3; background: var(--surface2); display: flex; align-items: center; justify-content: center; font-size: 1.3rem; }
.cast-name { font-size: 0.63rem; font-weight: 600; color: #bbb; padding: 5px 6px 2px; }
.cast-char { font-size: 0.56rem; color: var(--muted); padding: 0 6px 7px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ═══════════════════════════════════════════════
   FOR YOU / RECS
   ═══════════════════════════════════════════════ */
.rec-card {
  background: var(--surface); border-radius: 10px;
  border: 1px solid var(--border);
  padding: 14px 16px; margin-bottom: 10px;
  display: flex; gap: 14px; align-items: flex-start;
  transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s;
}
.rec-card:hover {
  border-color: rgba(201,169,110,0.28);
  transform: translateX(3px);
  box-shadow: 0 4px 22px rgba(0,0,0,0.4);
}
.rec-num { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 900; color: rgba(201,169,110,0.13); min-width: 34px; line-height: 1; text-align: right; }
.rec-poster { width: 50px; height: 75px; border-radius: 6px; object-fit: cover; flex-shrink: 0; }
.rec-poster-ph { width: 50px; height: 75px; border-radius: 6px; background: var(--surface2); display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0; }
.rec-body { flex: 1; min-width: 0; }
.rec-title { font-size: 0.86rem; font-weight: 600; color: var(--text); margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rec-pills { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 8px; }
.rec-pill { background: var(--accentdim); color: var(--accent2); font-size: 0.59rem; padding: 2px 7px; border-radius: 3px; border: 1px solid rgba(201,169,110,0.14); font-weight: 500; }
.rec-bar-bg { height: 2px; background: var(--subtle); border-radius: 2px; margin-bottom: 4px; }
.rec-bar { height: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 2px; }
.rec-score { font-size: 0.64rem; color: var(--accent); font-weight: 500; }

.chart-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }

/* ═══════════════════════════════════════════════
   WATCHLIST EMPTY STATE
   ═══════════════════════════════════════════════ */
.wl-empty { text-align: center; padding: 72px 40px; color: var(--muted); }
.wl-empty-icon { font-size: 2.8rem; margin-bottom: 14px; }
.wl-empty-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #fff; margin-bottom: 8px; }
.wl-empty-sub { font-size: 0.83rem; }

/* ═══════════════════════════════════════════════
   INPUTS
   ═══════════════════════════════════════════════ */
.stTextInput input {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 7px !important; font-size: 0.85rem !important;
  font-family: 'Outfit', sans-serif !important;
}
.stTextInput input:focus { border-color: rgba(201,169,110,0.4) !important; box-shadow: 0 0 0 2px rgba(201,169,110,0.07) !important; }
div[data-baseweb="select"] > div { background: var(--surface) !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 7px !important; }
.stCheckbox label { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 5px !important; padding: 5px 11px !important; font-size: 0.69rem !important; font-weight: 500 !important; color: var(--muted) !important; cursor: pointer !important; transition: border-color 0.15s, color 0.15s !important; white-space: nowrap !important; }
.stCheckbox label:hover { border-color: rgba(201,169,110,0.35) !important; color: var(--accent2) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TMDB_KEY = "e27947a52a97677530fdd9e476e8b17c"
IMG_BASE  = "https://image.tmdb.org/t/p"

GENRE_ICON = {
    'Action':'💥','Adventure':'🗺️','Animation':'🎨','Comedy':'😄',
    'Crime':'🕵️','Documentary':'📽️','Drama':'🎭','Fantasy':'🧙',
    'Horror':'👻','Musical':'🎵','Mystery':'🔍','Romance':'❤️',
    'Sci-Fi':'👽','Thriller':'🔪','War':'⚔️','Western':'🤠','Children':'🧒',
}

PROVIDER_LINKS = {
    'Netflix':            'https://www.netflix.com/search?q=',
    'Amazon Prime Video': 'https://www.amazon.com/s?k=',
    'Prime Video':        'https://www.amazon.com/s?k=',
    'Disney+':            'https://www.disneyplus.com/search/',
    'Hotstar':            'https://www.hotstar.com/in/search?q=',
    'Disney+ Hotstar':    'https://www.hotstar.com/in/search?q=',
    'Apple TV+':          'https://tv.apple.com/search?term=',
    'Hulu':               'https://www.hulu.com/search?q=',
    'Max':                'https://www.max.com/search?q=',
    'HBO Max':            'https://www.max.com/search?q=',
    'Peacock':            'https://www.peacocktv.com/search?q=',
    'Paramount+':         'https://www.paramountplus.com/search/',
    'Zee5':               'https://www.zee5.com/search/result/',
    'SonyLIV':            'https://www.sonyliv.com/search?q=',
    'Jio Cinema':         'https://www.jiocinema.com/search/',
}

def provider_href(name, title, fallback=''):
    base = PROVIDER_LINKS.get(name, '')
    return (base + urllib.parse.quote(title)) if base else (fallback or '#')

# ─────────────────────────────────────────────────────────────────────────────
# TMDB
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def tmdb_search(title):
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params={"api_key": TMDB_KEY, "query": clean}, timeout=5)
        if r.status_code != 200: return None
        res = r.json().get('results', [])
        wp  = [x for x in res if x.get('poster_path')]
        return wp[0] if wp else (res[0] if res else None)
    except Exception: return None

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
    except Exception: return None

@st.cache_data(show_spinner=False)
def poster_url(title):
    r = tmdb_search(title)
    return f"{IMG_BASE}/w300{r['poster_path']}" if r and r.get('poster_path') else None

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rp, mp = os.path.join("data","ratings.csv"), os.path.join("data","movies.csv")
    if not os.path.exists(rp) or not os.path.exists(mp): return None, None
    rd = pd.read_csv(rp)
    md = pd.read_csv(mp); md['genres'] = md['genres'].fillna('')
    return rd, md

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def genre_recs(picked, movies_df, n=12):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf.fit(movies_df['genres'])
    cosine = cosine_similarity(tfidf.transform(['|'.join(picked)]),
                               tfidf.transform(movies_df['genres'])).flatten()
    def cov(g): return len(set(picked) & {x.strip() for x in g.split('|')}) / len(picked)
    combined = 0.6 * cosine + 0.4 * movies_df['genres'].apply(cov).values
    mx = combined.max()
    if mx > 0: combined /= mx
    idx = combined.argsort()[::-1][:n]
    return pd.DataFrame([{'title': movies_df.iloc[i]['title'],
                           'genres': movies_df.iloc[i]['genres'],
                           'score': round(float(combined[i]), 4)} for i in idx])

def search_recs(query, movies_df, n=12):
    df = movies_df.copy()
    df['_t'] = df['title'].fillna('') + ' ' + df['genres'].fillna('').str.replace('|', ' ')
    tfidf  = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
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
for k, v in {'page': 'home', 'prev': 'home', 'movie': None,
              'genres': [], 'recs': None, 'watchlist': [],
              'rec_mode': 'genre', 'search_q': ''}.items():
    if k not in st.session_state:
        st.session_state[k] = v

ratings_df, movies_df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# NAV — pure Streamlit buttons inside a styled HTML shell
# ─────────────────────────────────────────────────────────────────────────────
wl_count = len(st.session_state.watchlist)
cur_page = st.session_state.page

st.markdown('<div class="nav-bar"><div class="nav-inner"><div class="nav-logo">CineMatch</div><div class="nav-sep"></div></div></div>', unsafe_allow_html=True)

# Render nav buttons via Streamlit columns — they actually work
nav_c1, nav_c2, nav_c3, nav_spacer = st.columns([1, 1, 1.4, 10])

with nav_c1:
    cls = 'nav-btn-active nav-btn-wrap' if cur_page == 'home' else 'nav-btn-wrap'
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button("Browse", key="nav_browse"):
        st.session_state.page = 'home'
        st.session_state.movie = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_c2:
    cls = 'nav-btn-active nav-btn-wrap' if cur_page == 'recs' else 'nav-btn-wrap'
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button("For You", key="nav_foryou"):
        st.session_state.page = 'recs'
        st.session_state.movie = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_c3:
    cls = 'nav-btn-active nav-btn-wrap' if cur_page == 'watchlist' else 'nav-btn-wrap'
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    wl_label = f"Watchlist ({wl_count})" if wl_count else "Watchlist"
    if st.button(wl_label, key="nav_watchlist"):
        st.session_state.page = 'watchlist'
        st.session_state.movie = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Move the nav row up into the nav-bar visually
st.markdown("""
<style>
/* Lift the nav columns row up into the sticky nav-bar */
section[data-testid="stMain"] > div > div:nth-child(2) > div[data-testid="stHorizontalBlock"] {
  position: relative;
  margin-top: -60px !important;
  background: rgba(8,9,14,0.97) !important;
  padding: 0 48px 0 220px !important;
  height: 60px !important;
  align-items: center !important;
  border-bottom: 1px solid rgba(255,255,255,0.07) !important;
  z-index: 999 !important;
}
section[data-testid="stMain"] > div > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  flex: 0 0 auto !important;
  width: auto !important;
  padding: 0 !important;
  min-width: 0 !important;
}
section[data-testid="stMain"] > div > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
  flex: 1 1 auto !important;
}
</style>
""", unsafe_allow_html=True)

if ratings_df is None:
    st.error("Data files not found. Please add data/ratings.csv and data/movies.csv.")
    st.stop()

all_genres = sorted({g.strip() for gs in movies_df['genres']
                     for g in gs.split('|') if g.strip() not in ('', '(no genres listed)')})

# ─────────────────────────────────────────────────────────────────────────────
# CARD GRID HELPER
# Invisible full-card button — NO visible symbol, NO text under card
# ─────────────────────────────────────────────────────────────────────────────
def render_grid(items, key_prefix, prev_page):
    """items: list of (title, genre1)"""
    cols = st.columns(8, gap="small")
    for i, (title, genre1) in enumerate(items):
        purl  = poster_url(title)
        icon  = GENRE_ICON.get(genre1, '🎬')
        short = title[:17] + '…' if len(title) > 17 else title

        with cols[i % 8]:
            img_html = (
                f'<img class="mcard-img" src="{purl}" loading="lazy" alt="{short}" '
                f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                f'<div class="mcard-ph" style="display:none;">{icon}</div>'
                if purl else f'<div class="mcard-ph">{icon}</div>'
            )
            # card HTML — no play symbol, no button text
            st.markdown(f"""
            <div class="card-col">
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
              </div>
            """, unsafe_allow_html=True)

            # Invisible button stretches over entire card via CSS
            if st.button("x", key=f"{key_prefix}_{i}", help=f"View {title}"):
                st.session_state.movie = title
                st.session_state.prev  = prev_page
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DETAIL PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_detail(title):
    # Back button
    st.markdown('<div style="max-width:1280px;margin:0 auto;padding:20px 48px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
    if st.button("← Back", key="back_btn"):
        st.session_state.movie = None
        st.session_state.page  = st.session_state.prev
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

    det = tmdb_details(title)
    if not det or 'title' not in det:
        st.warning("Could not load details from TMDB.")
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
        st.markdown(f'<div class="detail-backdrop"><img src="{backdrop}" alt=""/><div class="detail-fade"></div></div>',
                    unsafe_allow_html=True)

    pimg = (f'<img class="detail-poster" src="{poster}" alt="{ptitle}" '
            f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
            f'<div class="detail-poster-ph" style="display:none;">🎬</div>'
            if poster else '<div class="detail-poster-ph">🎬</div>')

    gpills = ''.join(f'<span class="detail-pill">{g}</span>' for g in genres)

    st.markdown(f"""
    <div class="detail-inner">
      <div class="detail-flex">
        {pimg}
        <div class="detail-info">
          <div class="detail-title">{ptitle}</div>
          <div class="detail-meta">
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

    # Watchlist action
    in_wl = title in st.session_state.watchlist
    st.markdown('<div style="max-width:1280px;margin:0 auto;padding:6px 48px 16px;display:flex;gap:10px;flex-wrap:wrap;">',
                unsafe_allow_html=True)
    if in_wl:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        if st.button("✓ In Watchlist — Remove", key="wl_tog"):
            st.session_state.watchlist.remove(title); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.button("＋ Add to Watchlist", key="wl_tog"):
            st.session_state.watchlist.append(title); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Trailer
    videos  = det.get('videos', {}).get('results', [])
    trailer = next((v for v in videos if v.get('type') == 'Trailer' and v.get('site') == 'YouTube'), None)
    if trailer:
        st.markdown(f'<div class="sub-section" style="border-top:none;padding-top:0;padding-bottom:16px;"><a class="trailer-btn" href="https://www.youtube.com/watch?v={trailer["key"]}" target="_blank">▶ &nbsp;Watch Trailer</a></div>',
                    unsafe_allow_html=True)

    # Where to Watch
    pd_data = det.get('watch/providers', {}).get('results', {})
    region  = pd_data.get('IN', pd_data.get('US', {}))
    jtw     = region.get('link', '')
    seen, combp = set(), []
    for label, lst in [('Stream', region.get('flatrate', [])),
                       ('Rent',   region.get('rent', [])),
                       ('Buy',    region.get('buy', []))]:
        for p in lst:
            if p['provider_name'] not in seen:
                seen.add(p['provider_name']); combp.append((p, label))

    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Where to Watch</div>', unsafe_allow_html=True)
    if combp:
        html_p = ''
        for p, label in combp[:10]:
            logo = f'{IMG_BASE}/w92{p["logo_path"]}' if p.get('logo_path') else None
            img  = f'<img src="{logo}" alt="{p["provider_name"]}"/>' if logo \
                   else '<div style="width:40px;height:40px;background:var(--surface2);border-radius:7px;"></div>'
            href = provider_href(p['provider_name'], ptitle, jtw)
            html_p += (f'<a class="provider" href="{href}" target="_blank" rel="noopener">'
                       f'{img}<div class="provider-name">{p["provider_name"]}</div>'
                       f'<div class="provider-type">{label}</div></a>')
        st.markdown(f'<div class="providers">{html_p}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:0.8rem;color:var(--muted);">No streaming data available for your region.</p>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Cast
    cast = det.get('credits', {}).get('cast', [])[:14]
    if cast:
        ch = ''
        for c in cast:
            img = f'{IMG_BASE}/w185{c["profile_path"]}' if c.get('profile_path') else None
            ph  = f'<img class="cast-img" src="{img}" alt="{c["name"]}"/>' if img else '<div class="cast-ph">👤</div>'
            ch += (f'<div class="cast-card">{ph}'
                   f'<div class="cast-name">{c["name"]}</div>'
                   f'<div class="cast-char">{c.get("character","")[:22]}</div></div>')
        st.markdown(f'<div class="sub-section"><div class="sub-title">Cast</div><div class="cast-grid">{ch}</div></div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.movie:
    show_detail(st.session_state.movie)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == 'home':

    # Hero
    st.markdown("""
    <div class="hero-band">
      <div class="hero-inner">
        <div class="hero-eyebrow">AI-Powered Discovery</div>
        <div class="hero-title">Your next favourite<br><em>film</em> awaits.</div>
        <div class="hero-sub">Browse thousands of movies or let our engine recommend
        films tailored to your taste — no account needed.</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Action row  — max-width container
    st.markdown('<div style="max-width:1280px;margin:0 auto;padding:24px 48px 0;display:flex;align-items:center;gap:14px;flex-wrap:wrap;">',
                unsafe_allow_html=True)
    a1, a2, _ = st.columns([1.6, 1.7, 9])
    with a1:
        if st.button("Get Recommendations →", key="hero_cta"):
            st.session_state.page = 'recs'; st.rerun()
    with a2:
        if st.session_state.watchlist:
            st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
            if st.button(f"🎯 My Watchlist ({wl_count})", key="wl_hero"):
                st.session_state.page = 'watchlist'; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider" style="margin-top:24px;"></div>', unsafe_allow_html=True)

    # Browse section
    st.markdown('<div style="max-width:1280px;margin:0 auto;padding:32px 48px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Explore</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Browse Movies</div>', unsafe_allow_html=True)

    f1, f2 = st.columns([3, 1])
    with f1:
        search = st.text_input("", placeholder="Search by title…",
                               label_visibility="collapsed", key="search_home")
    with f2:
        gpick = st.selectbox("", ['All'] + all_genres,
                             label_visibility="collapsed", key="gpick")

    if search:
        pool = movies_df[movies_df['title'].str.contains(search, case=False, na=False)]
    elif gpick != 'All':
        pool = movies_df[movies_df['genres'].str.contains(gpick, case=False, na=False)]
    else:
        pool = movies_df.sample(min(40, len(movies_df)), random_state=42)
    filtered = pool.head(40)

    st.markdown(f'<div class="count-line">{len(filtered)} TITLES</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Card grid — inside max-width wrapper
    st.markdown('<div style="max-width:1280px;margin:0 auto;padding:0 48px;">', unsafe_allow_html=True)
    items = [(row['title'], row['genres'].split('|')[0].strip() if row['genres'] else '')
             for _, row in filtered.iterrows()]
    render_grid(items, "hcard", "home")
    st.markdown('</div>', unsafe_allow_html=True)

    # Bottom banner
    st.markdown("""
    <div class="btm-band">
      <div class="btm-inner">
        <div>
          <div class="btm-title">Not sure what to watch?</div>
          <div class="btm-sub">Tell us your favourite genres or describe what you're in the mood for.</div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Find My Movies →", key="banner_cta"):
        st.session_state.page = 'recs'; st.rerun()
    st.markdown('</div></div><div style="height:48px;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == 'watchlist':

    st.markdown("""
    <div class="hero-band">
      <div class="hero-inner">
        <div class="hero-eyebrow">Your Collection</div>
        <div class="hero-title">My <em>Watchlist</em></div>
        <div class="hero-sub">Films you've saved to watch later.</div>
      </div>
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
        st.markdown('<div style="max-width:1280px;margin:0 auto;padding:32px 48px 0;">', unsafe_allow_html=True)
        st.markdown(f'<div class="count-line">{len(wl)} SAVED</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="max-width:1280px;margin:0 auto;padding:0 48px;">', unsafe_allow_html=True)
        wl_items = []
        for t in wl:
            gr     = movies_df[movies_df['title'] == t]
            genre1 = gr['genres'].values[0].split('|')[0].strip() if len(gr) else ''
            wl_items.append((t, genre1))
        render_grid(wl_items, "wlcard", "watchlist")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="max-width:1280px;margin:0 auto;padding:16px 48px 48px;">', unsafe_allow_html=True)
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("Clear Watchlist", key="clear_wl"):
            st.session_state.watchlist = []; st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOR YOU
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == 'recs':

    st.markdown("""
    <div class="hero-band">
      <div class="hero-inner">
        <div class="hero-eyebrow">Personalised</div>
        <div class="hero-title">Made <em>For You</em></div>
        <div class="hero-sub">Pick genres, or describe what you're in the mood for.</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="max-width:1280px;margin:0 auto;padding:32px 48px 0;">', unsafe_allow_html=True)

    # Mode toggle — two properly-sized side-by-side buttons
    st.markdown('<div class="section-eyebrow" style="margin-bottom:12px;">Recommendation Mode</div>',
                unsafe_allow_html=True)
    m1, m2, _ = st.columns([1.3, 1.5, 9])
    with m1:
        if st.button("🎭  By Genre", key="mode_genre"):
            st.session_state.rec_mode = 'genre'; st.session_state.recs = None
    with m2:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("🔍  By Search", key="mode_search"):
            st.session_state.rec_mode = 'search'; st.session_state.recs = None
        st.markdown('</div>', unsafe_allow_html=True)

    mode = st.session_state.rec_mode
    st.markdown(f'<p style="font-size:0.62rem;color:var(--accent);margin:10px 0 28px;letter-spacing:2px;font-weight:600;">{"GENRE-BASED" if mode == "genre" else "SEARCH-BASED"}</p>',
                unsafe_allow_html=True)

    # ── Genre mode ──
    if mode == 'genre':
        st.markdown('<div class="section-eyebrow" style="margin-bottom:12px;">Select genres you love</div>',
                    unsafe_allow_html=True)
        gcols  = st.columns(9)
        picked = []
        for idx, g in enumerate(all_genres):
            with gcols[idx % 9]:
                if st.checkbox(f"{GENRE_ICON.get(g, '🎬')} {g}", key=f"gc_{g}",
                               value=g in st.session_state.genres):
                    picked.append(g)
        st.session_state.genres = picked

        if picked:
            st.markdown(f'<p style="font-size:0.65rem;color:var(--accent2);margin:12px 0 2px;letter-spacing:1px;font-weight:500;">SELECTED: {" · ".join(picked)}</p>',
                        unsafe_allow_html=True)

        st.markdown('<div class="section-eyebrow" style="margin-top:24px;margin-bottom:10px;">Number of results</div>',
                    unsafe_allow_html=True)
        top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn")

        st.markdown('<div style="display:flex;gap:12px;margin-top:8px;">', unsafe_allow_html=True)
        fb1, fb2, _ = st.columns([1.2, 1.2, 9])
        with fb1:
            find_btn = st.button("Find Movies", key="find_btn")
        with fb2:
            st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
            if st.button("← Back to Browse", key="back_b"):
                st.session_state.page = 'home'; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if find_btn:
            if not picked:
                st.warning("Please select at least one genre.")
            else:
                with st.spinner("Finding movies…"):
                    st.session_state.recs = genre_recs(picked, movies_df, top_n)

    # ── Search mode ──
    else:
        st.markdown('<div class="section-eyebrow" style="margin-bottom:10px;">Describe what you want to watch</div>',
                    unsafe_allow_html=True)
        sq = st.text_input("", placeholder="e.g.  'space adventure with humour'  or  'dark crime thriller'",
                           label_visibility="collapsed", key="sq_input",
                           value=st.session_state.search_q)
        st.session_state.search_q = sq
        top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn_s")

        sb1, sb2, _ = st.columns([1.2, 1.5, 9])
        with sb1:
            srch_btn = st.button("Search Movies", key="srch_btn")
        with sb2:
            st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
            if st.button("← Back to Browse", key="back_bs"):
                st.session_state.page = 'home'; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if srch_btn:
            if not sq.strip():
                st.warning("Please enter a description.")
            else:
                with st.spinner("Searching…"):
                    st.session_state.recs = search_recs(sq, movies_df, top_n)

    # ── Results ──
    if st.session_state.recs is not None:
        recs = st.session_state.recs
        if recs.empty:
            st.warning("No results found. Try something different.")
        else:
            st.markdown('<div class="divider" style="margin:28px 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-eyebrow">Results</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-heading">Top {len(recs)} Picks For You</div>',
                        unsafe_allow_html=True)

            lc, rc = st.columns([1.2, 1])

            with lc:
                for i, row in recs.iterrows():
                    bw    = int(row['score'] * 100)
                    pills = ''.join(f'<span class="rec-pill">{g.strip()}</span>'
                                    for g in row['genres'].split('|') if g.strip())
                    purl  = poster_url(row['title'])
                    g0    = row['genres'].split('|')[0].strip()
                    icon  = GENRE_ICON.get(g0, '🎬')
                    ph_html = (
                        f'<img class="rec-poster" src="{purl}" alt="poster" '
                        f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                        f'<div class="rec-poster-ph" style="display:none;">{icon}</div>'
                        if purl else f'<div class="rec-poster-ph">{icon}</div>'
                    )
                    num = f"0{i+1}" if i + 1 < 10 else str(i + 1)

                    st.markdown(f"""
                    <div class="rec-card">
                      <div class="rec-num">{num}</div>
                      {ph_html}
                      <div class="rec-body">
                        <div class="rec-title">{row['title']}</div>
                        <div class="rec-pills">{pills}</div>
                        <div class="rec-bar-bg"><div class="rec-bar" style="width:{bw}%;"></div></div>
                        <div class="rec-score">Match: {row['score']:.2f}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="btn-text">', unsafe_allow_html=True)
                    if st.button(f"View details →  {row['title'][:32]}", key=f"rv_{i}"):
                        st.session_state.movie = row['title']
                        st.session_state.prev  = 'recs'
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            with rc:
                st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, max(4, len(recs) * 0.52)))
                fig.patch.set_facecolor('#0e0f18')
                ax.set_facecolor('#0e0f18')
                titles_ch = [r['title'][:26] + '…' if len(r['title']) > 26 else r['title']
                             for _, r in recs.iterrows()]
                scores = recs['score'].values
                pal    = (['#c9a96e','#cdb07c','#d1b78a','#d5be98','#d9c5a6',
                           '#ddcab0','#e1d0ba','#e5d5c4','#e9dace','#eddfd8'] * 2)[:len(scores)]
                ax.barh(titles_ch[::-1], scores[::-1], color=pal[::-1], height=0.5, edgecolor='none')
                for b, s in zip(ax.patches, scores[::-1]):
                    ax.text(b.get_width() + 0.014, b.get_y() + b.get_height() / 2,
                            f'{s:.2f}', va='center', color='#5a5a72', fontsize=7.5)
                ax.set_xlim(0, 1.22)
                ax.set_xlabel('Match Score', color='#3a3a52', fontsize=8)
                ax.set_title('Match Chart', color='#c9a96e', fontsize=9, fontweight='bold', pad=10)
                ax.tick_params(colors='#3a3a52', labelsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor('#1e1e2e')
                plt.tight_layout(pad=1.2)
                st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div style="margin-top:12px;">', unsafe_allow_html=True)
                st.download_button("⬇ Download CSV", recs.to_csv(index=False),
                                   "recommendations.csv", "text/csv", key="dl_csv")
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div><div style="height:48px;"></div>', unsafe_allow_html=True)
