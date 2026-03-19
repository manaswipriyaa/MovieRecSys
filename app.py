import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, requests

st.set_page_config(page_title="MovieRecSys", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #060714; color: #e8e8f0; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stSidebar"] { display: none !important; }
footer, header { display: none !important; }

/* ── TOPBAR ── */
.topbar {
    width: 100%; height: 60px;
    background: rgba(6,7,20,0.97);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    display: flex; align-items: center;
    padding: 0 40px; gap: 28px;
    position: sticky; top: 0; z-index: 100;
}
.topbar-logo {
    font-family: 'Bebas Neue', cursive; font-size: 1.7rem; letter-spacing: 4px;
    background: linear-gradient(90deg,#7c6dfa,#e040fb,#ff4081);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    flex-shrink: 0; margin-right: 4px;
}

/* ── NAV BUTTONS in topbar ── */
div[data-testid="stHorizontalBlock"]:first-of-type {
    position: relative; margin-top: -60px; padding: 0 40px 0 220px;
    height: 60px; display: flex; align-items: center; gap: 6px;
    pointer-events: none; z-index: 200;
}
div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] {
    flex: 0 !important; min-width: 0 !important; width: auto !important;
    pointer-events: all;
}
div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:last-child {
    flex: 1 !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    background: transparent !important;
    color: #888 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    padding: 5px 20px !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    width: auto !important; min-width: 0 !important;
    transition: all 0.2s !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
    background: rgba(124,109,250,0.12) !important;
    color: white !important;
    border-color: rgba(124,109,250,0.4) !important;
    transform: none !important;
}

/* ── HERO ── */
.hero {
    position: relative; width: 100%; height: 360px; overflow: hidden;
    background: linear-gradient(135deg,#0d0b2b 0%,#1a0533 55%,#0a1628 100%);
    display: flex; align-items: flex-end; padding: 0 52px 44px;
}
.hero-glow {
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 55% 55% at 72% 38%, rgba(124,109,250,0.16) 0%, transparent 60%),
                radial-gradient(ellipse 35% 35% at 28% 65%, rgba(224,64,251,0.1) 0%, transparent 60%);
}
.hero-fade { position: absolute; bottom: 0; left: 0; right: 0; height: 160px; background: linear-gradient(0deg,#060714,transparent); }
.hero-content { position: relative; z-index: 2; max-width: 560px; }
.hero-badge {
    display: inline-block; background: rgba(124,109,250,0.18); color: #a78bfa;
    font-size: 0.68rem; letter-spacing: 2.5px; text-transform: uppercase;
    padding: 4px 14px; border-radius: 50px; border: 1px solid rgba(124,109,250,0.3);
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Bebas Neue', cursive; font-size: 3rem; letter-spacing: 3px;
    color: white; line-height: 0.95; margin-bottom: 14px;
}
.hero-title span { background: linear-gradient(90deg,#7c6dfa,#e040fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-desc { font-size: 0.88rem; color: #8080a0; line-height: 1.65; margin-bottom: 22px; font-weight: 300; }

/* Hero CTA button */
.hero-cta-row { padding: 0 52px; margin-top: -14px; margin-bottom: 32px; position: relative; z-index: 10; }
.hero-cta-row .stButton > button {
    background: linear-gradient(135deg,#7c6dfa,#e040fb) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    padding: 11px 28px !important;
    box-shadow: 0 6px 24px rgba(124,109,250,0.35) !important;
    width: auto !important; min-width: 0 !important;
}

/* ── SECTION ── */
.section { padding: 36px 44px; }
.section-label { font-size: 0.65rem; letter-spacing: 3px; text-transform: uppercase; color: #7c6dfa; font-weight: 600; margin-bottom: 5px; }
.section-title { font-family: 'Bebas Neue', cursive; font-size: 1.6rem; letter-spacing: 2px; color: white; margin-bottom: 22px; }

/* ── MOVIE CARDS ── */
.mcard {
    border-radius: 9px; overflow: hidden; background: #10111f;
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.22s, box-shadow 0.22s, border-color 0.22s;
    cursor: pointer; position: relative;
}
.mcard:hover { transform: scale(1.05) translateY(-4px); box-shadow: 0 14px 40px rgba(0,0,0,0.65); border-color: rgba(124,109,250,0.4); }
.mcard:hover .mcard-overlay { opacity: 1; }
.mcard-poster { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.mcard-poster-ph { width: 100%; aspect-ratio: 2/3; background: linear-gradient(135deg,#12132a,#1e1040); display: flex; align-items: center; justify-content: center; font-size: 2.2rem; }
.mcard-overlay {
    position: absolute; inset: 0; background: rgba(124,109,250,0.18);
    display: flex; align-items: center; justify-content: center;
    opacity: 0; transition: opacity 0.2s;
}
.mcard-overlay-icon { font-size: 2rem; }
.mcard-info { padding: 8px 9px 11px; }
.mcard-title { font-size: 0.74rem; font-weight: 500; color: #ddd; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }
.mcard-genre { font-size: 0.62rem; color: #444; }

/* Invisible click overlay button on each card */
.mcard-btn-wrap { position: relative; }
.mcard-btn-wrap .stButton { position: absolute; inset: 0; z-index: 5; }
.mcard-btn-wrap .stButton > button {
    position: absolute !important; inset: 0 !important;
    width: 100% !important; height: 100% !important;
    background: transparent !important; border: none !important;
    box-shadow: none !important; color: transparent !important;
    font-size: 0 !important; padding: 0 !important;
    cursor: pointer !important; border-radius: 9px !important;
}
.mcard-btn-wrap .stButton > button:hover { background: transparent !important; transform: none !important; }

/* ── BOTTOM CTA STRIP ── */
.cta-strip {
    background: linear-gradient(135deg,#0d0b2b,#160428);
    padding: 44px 52px; margin-top: 40px;
    display: flex; align-items: center; justify-content: space-between; gap: 40px;
}
.cta-strip-text {}
.cta-strip-title { font-family: 'Bebas Neue', cursive; font-size: 1.8rem; letter-spacing: 3px; color: white; margin-bottom: 7px; }
.cta-strip-sub { font-size: 0.83rem; color: #555; font-weight: 300; }
.cta-strip-btn .stButton > button {
    background: linear-gradient(135deg,#7c6dfa,#e040fb) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    padding: 11px 30px !important; white-space: nowrap !important;
    box-shadow: 0 6px 24px rgba(124,109,250,0.3) !important;
    width: auto !important; min-width: 0 !important;
}

/* ── FOR YOU PAGE ── */
.foryou-hero {
    position: relative; width: 100%; height: 240px; overflow: hidden;
    background: linear-gradient(135deg,#0d0b2b,#1a0533,#0a1628);
    display: flex; align-items: flex-end; padding: 0 52px 36px;
}
.rec-card {
    background: #10111f; border-radius: 13px; padding: 16px 18px;
    margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.06);
    display: flex; gap: 14px; align-items: flex-start; cursor: pointer;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: rgba(124,109,250,0.3); }
.rec-num { font-family: 'Bebas Neue', cursive; font-size: 2rem; color: rgba(124,109,250,0.18); min-width: 38px; text-align: right; line-height: 1; }
.rec-poster { width: 50px; height: 74px; border-radius: 7px; object-fit: cover; flex-shrink: 0; background: #1a1b30; }
.rec-body { flex: 1; padding-top: 1px; }
.rec-title { font-size: 0.9rem; font-weight: 600; color: #eee; margin-bottom: 5px; }
.rec-pills { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.rec-pill { background: rgba(124,109,250,0.1); color: #a78bfa; font-size: 0.62rem; padding: 2px 7px; border-radius: 50px; border: 1px solid rgba(124,109,250,0.18); }
.rec-bar-bg { background: rgba(255,255,255,0.06); border-radius: 50px; height: 3px; }
.rec-bar { background: linear-gradient(90deg,#7c6dfa,#e040fb); border-radius: 50px; height: 3px; }
.rec-score { font-size: 0.72rem; color: #7c6dfa; margin-top: 4px; }

/* ── DETAIL PAGE ── */
.detail-backdrop { width: 100%; height: 240px; overflow: hidden; position: relative; }
.detail-backdrop img { width: 100%; height: 100%; object-fit: cover; opacity: 0.3; }
.detail-backdrop-fade { position: absolute; bottom: 0; left: 0; right: 0; height: 130px; background: linear-gradient(0deg,#060714,transparent); }
.detail-hero { display: flex; gap: 36px; padding: 32px 48px 36px; align-items: flex-start; background: #060714; }
.detail-poster { width: 200px; min-width: 200px; border-radius: 12px; box-shadow: 0 18px 50px rgba(0,0,0,0.7); }
.detail-poster-ph { width: 200px; min-width: 200px; height: 300px; border-radius: 12px; background: #1a1b30; display: flex; align-items: center; justify-content: center; font-size: 3.5rem; }
.detail-info { flex: 1; }
.detail-title { font-family: 'Bebas Neue', cursive; font-size: 2.8rem; letter-spacing: 2px; color: white; margin-bottom: 10px; line-height: 1; }
.detail-meta { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 13px; }
.detail-rating { background: rgba(124,109,250,0.15); color: #a78bfa; font-size: 0.78rem; font-weight: 600; padding: 3px 10px; border-radius: 50px; border: 1px solid rgba(124,109,250,0.25); }
.detail-year, .detail-runtime { font-size: 0.8rem; color: #555; }
.detail-tagline { font-style: italic; color: #444; font-size: 0.85rem; margin-bottom: 13px; }
.detail-overview { font-size: 0.87rem; color: #999; line-height: 1.75; max-width: 620px; margin-bottom: 18px; }
.detail-genre-pill { display: inline-block; background: rgba(255,255,255,0.05); color: #bbb; font-size: 0.72rem; padding: 4px 13px; border-radius: 50px; border: 1px solid rgba(255,255,255,0.09); margin-right: 6px; margin-bottom: 6px; }
.watch-section, .cast-section { padding: 28px 48px; border-top: 1px solid rgba(255,255,255,0.05); }
.section-h { font-family: 'Bebas Neue', cursive; font-size: 1.3rem; letter-spacing: 2px; color: white; margin-bottom: 18px; }
.provider-row { display: flex; gap: 12px; flex-wrap: wrap; }
.provider-card { background: #10111f; border-radius: 10px; padding: 12px 16px; border: 1px solid rgba(255,255,255,0.07); display: flex; flex-direction: column; align-items: center; gap: 5px; min-width: 90px; }
.provider-logo { width: 44px; height: 44px; border-radius: 8px; object-fit: cover; }
.provider-name { font-size: 0.68rem; color: #888; text-align: center; }
.provider-type { font-size: 0.6rem; color: #555; }
.cast-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px,1fr)); gap: 12px; }
.cast-card { background: #10111f; border-radius: 9px; overflow: hidden; border: 1px solid rgba(255,255,255,0.04); }
.cast-photo { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; background: #1a1b30; }
.cast-ph { width: 100%; aspect-ratio: 2/3; background: #1a1b30; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; }
.cast-info { padding: 7px 8px 9px; }
.cast-name { font-size: 0.7rem; font-weight: 600; color: #ccc; margin-bottom: 2px; }
.cast-char { font-size: 0.62rem; color: #444; }
.back-wrap { padding: 14px 48px 0; }
.back-wrap .stButton > button {
    background: rgba(255,255,255,0.05) !important; color: #888 !important;
    border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 50px !important;
    font-size: 0.72rem !important; letter-spacing: 1px !important;
    padding: 5px 18px !important; box-shadow: none !important;
    width: auto !important; min-width: 0 !important;
}

/* ── GLOBAL BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg,#7c6dfa,#e040fb) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    padding: 10px 26px !important;
    box-shadow: 0 5px 20px rgba(124,109,250,0.3) !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stTextInput > div > div > input { background: #10111f !important; color: #eee !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
div[data-baseweb="select"] > div { background: #10111f !important; border-color: rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── TMDB ──────────────────────────────────────────────────────────────────────
TMDB_KEY = "e27947a52a97677530fdd9e476e8b17c"
TMDB_IMG = "https://image.tmdb.org/t/p/w300"

@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params={"api_key": TMDB_KEY, "query": clean}, timeout=4)
        res = r.json().get('results', [])
        if res and res[0].get('poster_path'):
            return TMDB_IMG + res[0]['poster_path']
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def fetch_movie_details(title):
    try:
        clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params={"api_key": TMDB_KEY, "query": clean}, timeout=5)
        res = r.json().get('results', [])
        if not res:
            return None
        mid = res[0]['id']
        det = requests.get(f"https://api.themoviedb.org/3/movie/{mid}",
                           params={"api_key": TMDB_KEY,
                                   "append_to_response": "credits,watch/providers,videos"},
                           timeout=5).json()
        return det
    except Exception:
        return None

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rp, mp = os.path.join("data","ratings.csv"), os.path.join("data","movies.csv")
    if not os.path.exists(rp) or not os.path.exists(mp):
        return None, None
    rd = pd.read_csv(rp); md = pd.read_csv(mp)
    md['genres'] = md['genres'].fillna('')
    return rd, md

@st.cache_data
def build_tfidf(movies_df):
    t = TfidfVectorizer(token_pattern=r'[^|]+')
    m = t.fit_transform(movies_df['genres'])
    return t, m

def get_genre_recs(picked, movies_df, top_n=12):
    t = TfidfVectorizer(token_pattern=r'[^|]+')
    t.fit(movies_df['genres'])
    qv = t.transform(['|'.join(picked)])
    mv = t.transform(movies_df['genres'])
    sims = cosine_similarity(qv, mv).flatten()
    idx  = sims.argsort()[::-1][:top_n]
    return pd.DataFrame([{'title': movies_df.iloc[i]['title'],
                          'genres': movies_df.iloc[i]['genres'],
                          'score': round(float(sims[i]), 4)} for i in idx])

GENRE_ICON = {
    'Action':'💥','Adventure':'🗺️','Animation':'🎨','Comedy':'😂','Crime':'🕵️',
    'Documentary':'📽️','Drama':'🎭','Fantasy':'🧙','Horror':'👻','Musical':'🎵',
    'Mystery':'🔍','Romance':'❤️','Sci-Fi':'👽','Thriller':'🔪','War':'⚔️',
    'Western':'🤠','Children':'🧒',
}

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [('page','home'),('sel_genres',[]),('recs_ready',False),
             ('recs_df',None),('sel_movie',None),('prev_page','home')]:
    if k not in st.session_state:
        st.session_state[k] = v

ratings_df, movies_df = load_data()

# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown('<div class="topbar"><div class="topbar-logo">MOVIERECSYS</div></div>',
            unsafe_allow_html=True)

# Nav buttons float up into topbar via CSS negative margin
n1, n2, _spacer = st.columns([1, 1, 14])
with n1:
    if st.button("Browse", key="nav_home"):
        st.session_state.sel_movie = None
        st.session_state.page = 'home'
        st.rerun()
with n2:
    if st.button("For You", key="nav_recs"):
        st.session_state.sel_movie = None
        st.session_state.page = 'recs'
        st.rerun()

if ratings_df is None:
    st.error("Data files not found! Add data/ratings.csv and data/movies.csv")
    st.stop()

all_genres = sorted({g.strip() for gs in movies_df['genres']
                     for g in gs.split('|')
                     if g.strip() and g.strip() != '(no genres listed)'})

# ── DETAIL PAGE ───────────────────────────────────────────────────────────────
def render_detail(title):
    st.markdown('<div class="back-wrap">', unsafe_allow_html=True)
    if st.button("← Back", key="back"):
        st.session_state.sel_movie = None
        st.session_state.page = st.session_state.prev_page
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    det = fetch_movie_details(title)
    if not det or 'title' not in det:
        st.error("Couldn't load movie details.")
        return

    poster   = f"https://image.tmdb.org/t/p/w400{det['poster_path']}"   if det.get('poster_path')   else None
    backdrop = f"https://image.tmdb.org/t/p/w1280{det['backdrop_path']}" if det.get('backdrop_path') else None
    rating   = round(det.get('vote_average', 0), 1)
    year     = det.get('release_date', '')[:4]
    rt       = det.get('runtime', 0)
    runtime  = f"{rt//60}h {rt%60}m" if rt else ''
    genres   = [g['name'] for g in det.get('genres', [])]
    tagline  = det.get('tagline', '')
    overview = det.get('overview', 'No description available.')

    if backdrop:
        st.markdown(f'''<div class="detail-backdrop">
            <img src="{backdrop}"/>
            <div class="detail-backdrop-fade"></div>
        </div>''', unsafe_allow_html=True)

    poster_html = f'<img class="detail-poster" src="{poster}"/>' if poster \
                  else '<div class="detail-poster-ph">🎬</div>'
    genre_pills = ''.join([f'<span class="detail-genre-pill">{g}</span>' for g in genres])

    st.markdown(f'''
    <div class="detail-hero">
        {poster_html}
        <div class="detail-info">
            <div class="detail-title">{det["title"]}</div>
            <div class="detail-meta">
                <span class="detail-year">{year}</span>
                <span class="detail-rating">⭐ {rating} / 10</span>
                <span class="detail-runtime">{runtime}</span>
            </div>
            {f\'<div class="detail-tagline">"{tagline}"</div>\' if tagline else ""}
            <div class="detail-overview">{overview}</div>
            <div>{genre_pills}</div>
        </div>
    </div>''', unsafe_allow_html=True)

    # Trailer
    videos  = det.get('videos', {}).get('results', [])
    trailer = next((v for v in videos if v['type'] == 'Trailer' and v['site'] == 'YouTube'), None)
    if trailer:
        st.markdown(f'''
        <div style="padding:20px 48px 0;">
            <a href="https://www.youtube.com/watch?v={trailer["key"]}" target="_blank"
               style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,0,0,0.12);
               color:#ff5555;font-size:0.78rem;font-weight:600;letter-spacing:1px;
               padding:9px 20px;border-radius:50px;border:1px solid rgba(255,0,0,0.25);text-decoration:none;">
               ▶ &nbsp;Watch Trailer on YouTube
            </a>
        </div>''', unsafe_allow_html=True)

    # Where to watch
    providers_raw = det.get('watch/providers', {}).get('results', {})
    region = providers_raw.get('IN', providers_raw.get('US', {}))
    flat   = region.get('flatrate', [])
    rent   = region.get('rent',     [])
    buy    = region.get('buy',      [])
    seen_p = set()
    all_p  = []
    for ptype, plist in [('Stream', flat), ('Rent', rent), ('Buy', buy)]:
        for p in plist:
            if p['provider_name'] not in seen_p:
                seen_p.add(p['provider_name'])
                all_p.append((p, ptype))

    if all_p:
        cards = ''
        for p, ptype in all_p[:12]:
            logo = f"https://image.tmdb.org/t/p/w92{p['logo_path']}" if p.get('logo_path') else None
            logo_html = f'<img class="provider-logo" src="{logo}"/>' if logo \
                        else '<div class="provider-logo" style="background:#1a1b30;border-radius:8px;"></div>'
            cards += f'''<div class="provider-card">{logo_html}
                <div class="provider-name">{p["provider_name"]}</div>
                <div class="provider-type">{ptype}</div></div>'''
        st.markdown(f'''<div class="watch-section">
            <div class="section-h">Where to Watch</div>
            <div class="provider-row">{cards}</div></div>''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="watch-section"><div class="section-h">Where to Watch</div>'
                    '<p style="color:#444;font-size:0.83rem;">No streaming data available for your region.</p></div>',
                    unsafe_allow_html=True)

    # Cast
    cast = det.get('credits', {}).get('cast', [])[:14]
    if cast:
        cast_cards = ''
        for c in cast:
            ph  = f"https://image.tmdb.org/t/p/w185{c['profile_path']}" if c.get('profile_path') else None
            phi = f'<img class="cast-photo" src="{ph}"/>' if ph else '<div class="cast-ph">👤</div>'
            cast_cards += f'''<div class="cast-card">{phi}
                <div class="cast-info">
                    <div class="cast-name">{c["name"]}</div>
                    <div class="cast-char">{c.get("character","")[:22]}</div>
                </div></div>'''
        st.markdown(f'''<div class="cast-section">
            <div class="section-h">Cast</div>
            <div class="cast-grid">{cast_cards}</div></div>''', unsafe_allow_html=True)

# ── Route to detail page ──────────────────────────────────────────────────────
if st.session_state.sel_movie:
    render_detail(st.session_state.sel_movie)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':

    st.markdown('''
    <div class="hero">
        <div class="hero-glow"></div><div class="hero-fade"></div>
        <div class="hero-content">
            <div class="hero-badge">Powered by AI</div>
            <div class="hero-title">Discover<br><span>Cinema</span><br>You\'ll Love</div>
            <div class="hero-desc">Browse thousands of movies or let our AI recommend<br>films based on your taste — no account needed.</div>
        </div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="hero-cta-row">', unsafe_allow_html=True)
    if st.button("Get Personalised Recs", key="hero_cta"):
        st.session_state.page = 'recs'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Explore</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Browse Movies</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        search = st.text_input("", placeholder="Search a movie title...", label_visibility="collapsed")
    with sc2:
        gpick = st.selectbox("", ['All'] + all_genres, label_visibility="collapsed")

    if search:
        filtered = movies_df[movies_df['title'].str.contains(search, case=False, na=False)].head(40)
    elif gpick != 'All':
        pool = movies_df[movies_df['genres'].str.contains(gpick, case=False, na=False)]
        filtered = pool.sample(min(40, len(pool)), random_state=1)
    else:
        filtered = movies_df.sample(min(40, len(movies_df)), random_state=42)

    st.markdown(f'<div style="font-size:0.72rem;color:#444;margin-bottom:18px;letter-spacing:1px;">{len(filtered)} TITLES</div>',
                unsafe_allow_html=True)

    cols = st.columns(8)
    for idx, (_, row) in enumerate(filtered.head(40).iterrows()):
        short  = row['title'][:17] + ('…' if len(row['title']) > 17 else '')
        genre1 = row['genres'].split('|')[0].strip() if row['genres'] else ''
        purl   = fetch_poster(row['title'])
        with cols[idx % 8]:
            # Render card HTML
            if purl:
                st.markdown(f'''<div class="mcard">
                    <img class="mcard-poster" src="{purl}" loading="lazy"/>
                    <div class="mcard-overlay"><span class="mcard-overlay-icon">▶</span></div>
                    <div class="mcard-info">
                        <div class="mcard-title">{short}</div>
                        <div class="mcard-genre">{genre1}</div>
                    </div></div>''', unsafe_allow_html=True)
            else:
                icon = GENRE_ICON.get(genre1, '🎬')
                st.markdown(f'''<div class="mcard">
                    <div class="mcard-poster-ph">{icon}</div>
                    <div class="mcard-overlay"><span class="mcard-overlay-icon">▶</span></div>
                    <div class="mcard-info">
                        <div class="mcard-title">{short}</div>
                        <div class="mcard-genre">{genre1}</div>
                    </div></div>''', unsafe_allow_html=True)
            # Transparent click button
            if st.button("", key=f"m{idx}", help=row['title']):
                st.session_state.sel_movie  = row['title']
                st.session_state.prev_page = 'home'
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Bottom CTA strip
    st.markdown('''
    <div class="cta-strip">
        <div class="cta-strip-text">
            <div class="cta-strip-title">Not sure what to watch?</div>
            <div class="cta-strip-sub">Tell us your favourite genres — we\'ll pick the perfect movies.</div>
        </div>
        <div class="cta-strip-btn" id="cta-strip-slot"></div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div style="display:flex;justify-content:flex-end;padding:0 52px;margin-top:-62px;position:relative;z-index:10;">',
                unsafe_allow_html=True)
    if st.button("Find My Movies", key="bottom_cta"):
        st.session_state.page = 'recs'
        st.rerun()
    st.markdown('</div><div style="background:linear-gradient(135deg,#0d0b2b,#160428);height:20px;"></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOR YOU PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'recs':

    st.markdown('''
    <div class="foryou-hero">
        <div class="hero-glow"></div><div class="hero-fade"></div>
        <div class="hero-content">
            <div class="hero-badge">Personalised</div>
            <div class="hero-title" style="font-size:2.6rem;">Made <span>For You</span></div>
            <div class="hero-desc">Pick your favourite genres and get instant recommendations.</div>
        </div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 1 — Pick genres</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What do you like to watch?</div>', unsafe_allow_html=True)

    gcols  = st.columns(9)
    picked = []
    for i, g in enumerate(all_genres):
        icon = GENRE_ICON.get(g, '🎬')
        with gcols[i % 9]:
            if st.checkbox(f"{icon} {g}", key=f"g_{g}", value=(g in st.session_state.sel_genres)):
                picked.append(g)
    st.session_state.sel_genres = picked

    if picked:
        st.markdown(f'<div style="color:#7c6dfa;font-size:0.75rem;margin:14px 0 6px;letter-spacing:1px;">SELECTED: {" · ".join(picked)}</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:20px;">Step 2 — How many?</div>',
                unsafe_allow_html=True)
    top_n = st.slider("", 5, 20, 10, label_visibility="collapsed")

    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        find = st.button("Find My Movies", key="find")
    with b2:
        if st.button("Browse All", key="back_browse"):
            st.session_state.page = 'home'
            st.rerun()

    if find:
        if not picked:
            st.warning("Please select at least one genre.")
        else:
            with st.spinner("Finding your movies..."):
                recs = get_genre_recs(picked, movies_df, top_n)
            st.session_state.recs_df    = recs
            st.session_state.recs_ready = True

    if st.session_state.recs_ready and st.session_state.recs_df is not None:
        recs = st.session_state.recs_df
        st.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:28px 0;"/>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">Your Results</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Top {len(recs)} Picks For You</div>', unsafe_allow_html=True)

        lc, rc = st.columns([1.1, 1])
        with lc:
            for i, row in recs.iterrows():
                bw     = int(row['score'] * 100)
                pills  = ''.join([f'<span class="rec-pill">{g.strip()}</span>'
                                  for g in row['genres'].split('|') if g.strip()])
                poster = fetch_poster(row['title'])
                ph     = f'<img class="rec-poster" src="{poster}"/>' if poster \
                         else f'<div class="rec-poster" style="display:flex;align-items:center;justify-content:center;font-size:1.4rem;">{GENRE_ICON.get(row["genres"].split("|")[0].strip(),"🎬")}</div>'
                st.markdown(f'''<div class="rec-card">
                    <div class="rec-num">0{i+1}</div>{ph}
                    <div class="rec-body">
                        <div class="rec-title">{row["title"]}</div>
                        <div class="rec-pills">{pills}</div>
                        <div class="rec-bar-bg"><div class="rec-bar" style="width:{bw}%"></div></div>
                        <div class="rec-score">Match: {row["score"]}</div>
                    </div></div>''', unsafe_allow_html=True)
                if st.button("View Details", key=f"rd{i}"):
                    st.session_state.sel_movie  = row['title']
                    st.session_state.prev_page = 'recs'
                    st.rerun()

        with rc:
            fig, ax = plt.subplots(figsize=(6, max(4, len(recs) * 0.52)))
            fig.patch.set_facecolor('#060714')
            ax.set_facecolor('#10111f')
            titles = [r['title'][:22] + ('…' if len(r['title']) > 22 else '') for _, r in recs.iterrows()]
            scores = recs['score'].values
            pal = ['#7c6dfa','#a78bfa','#e040fb','#ff4081','#00e5ff',
                   '#69f0ae','#ffea00','#ff6d00','#7c6dfa','#a78bfa',
                   '#e040fb','#ff4081','#00e5ff','#69f0ae','#ffea00',
                   '#ff6d00','#7c6dfa','#a78bfa','#e040fb','#ff4081']
            bars = ax.barh(titles[::-1], scores[::-1], color=pal[:len(scores)][::-1], height=0.56)
            for b in bars:
                ax.text(b.get_width()+0.004, b.get_y()+b.get_height()/2,
                        f'{b.get_width():.2f}', va='center', color='#666', fontsize=8)
            ax.set_xlabel('Match Score', color='#444', fontsize=8)
            ax.set_title('Your Match Chart', color='#ddd', fontsize=11, fontweight='bold', pad=12)
            ax.tick_params(colors='#555', labelsize=7.5)
            for s in ax.spines.values(): s.set_edgecolor('#1a1a1a')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            st.markdown('<br>', unsafe_allow_html=True)
            st.download_button("Download CSV", recs.to_csv(index=False), "recs.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)
