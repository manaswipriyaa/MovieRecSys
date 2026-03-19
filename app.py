import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, requests

st.set_page_config(
    page_title="MovieRecSys",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #07080f; color: #e0e0e8; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stSidebar"], footer, header { display: none !important; }

/* NAV */
.nav {
    height: 56px; background: #07080f;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: center;
    padding: 0 48px; gap: 32px;
    position: sticky; top: 0; z-index: 999;
}
.nav-logo {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.6rem; letter-spacing: 5px;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; flex-shrink: 0;
}
.nav-divider { width: 1px; height: 20px; background: rgba(255,255,255,0.1); }

/* NAV BUTTONS — float into the nav bar */
div[data-testid="stHorizontalBlock"]:first-of-type {
    background: #07080f !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    position: sticky !important; top: 0 !important; z-index: 998 !important;
    margin-top: -57px !important;
    height: 56px !important;
    padding: 0 48px 0 260px !important;
    display: flex !important; align-items: center !important; gap: 4px !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] {
    flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
    padding: 0 !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:last-child {
    flex: 1 1 auto !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    background: transparent !important; color: #666 !important;
    border: none !important; box-shadow: none !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    padding: 6px 14px !important; border-radius: 6px !important;
    width: auto !important; min-width: 0 !important;
    transition: color 0.15s, background 0.15s !important;
    transform: none !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
    color: #e0e0e8 !important;
    background: rgba(255,255,255,0.06) !important;
    transform: none !important;
}

/* HERO */
.hero {
    width: 100%; padding: 72px 48px 64px;
    background: linear-gradient(160deg, #0e0b24 0%, #130824 50%, #080d1e 100%);
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute;
    top: -100px; right: -100px; width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(129,140,248,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: ''; position: absolute;
    bottom: -60px; left: 10%; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(192,132,252,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.hero-label {
    font-size: 0.65rem; letter-spacing: 3px; text-transform: uppercase;
    color: #818cf8; font-weight: 600; margin-bottom: 18px;
}
.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 4rem; letter-spacing: 2px;
    color: #fff; line-height: 1; margin-bottom: 16px;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 0.9rem; color: #6b6b80; line-height: 1.7;
    max-width: 480px; margin-bottom: 0; font-weight: 300;
}

/* CTA BUTTONS */
.cta-row { padding: 28px 48px 0; display: flex; gap: 12px; align-items: center; }
.cta-row .stButton > button {
    background: linear-gradient(135deg, #818cf8, #c084fc) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-size: 0.78rem !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    text-transform: uppercase !important; padding: 11px 24px !important;
    box-shadow: 0 4px 20px rgba(129,140,248,0.25) !important;
    width: auto !important; min-width: 0 !important;
}

/* DIVIDER */
.divider { height: 1px; background: rgba(255,255,255,0.05); margin: 0; }

/* BROWSE SECTION */
.browse-section { padding: 40px 48px; }
.section-eyebrow {
    font-size: 0.62rem; letter-spacing: 3px; text-transform: uppercase;
    color: #818cf8; font-weight: 600; margin-bottom: 6px;
}
.section-heading {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.5rem; letter-spacing: 2px; color: #fff; margin-bottom: 24px;
}

/* MOVIE CARD */
.mcard {
    border-radius: 8px; overflow: hidden;
    background: #0f1018;
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative; cursor: pointer;
}
.mcard:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.6);
    border-color: rgba(129,140,248,0.3);
}
.mcard-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.mcard-placeholder {
    width: 100%; aspect-ratio: 2/3;
    background: linear-gradient(160deg, #0f1018, #1a1830);
    display: flex; align-items: center; justify-content: center; font-size: 2.4rem;
}
.mcard-hover {
    position: absolute; inset: 0;
    background: rgba(129,140,248,0.15);
    display: flex; align-items: center; justify-content: center;
    opacity: 0; transition: opacity 0.2s;
    font-size: 1.6rem;
}
.mcard:hover .mcard-hover { opacity: 1; }
.mcard-body { padding: 9px 10px 11px; }
.mcard-title {
    font-size: 0.73rem; font-weight: 500; color: #ccc;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px;
}
.mcard-genre { font-size: 0.62rem; color: #3d3d50; }

/* invisible click button on each card */
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
.mcard-wrap .stButton > button:hover {
    background: transparent !important; transform: none !important;
}

/* BOTTOM BANNER */
.bottom-banner {
    margin: 40px 48px;
    background: linear-gradient(135deg, #0e0b24, #130824);
    border: 1px solid rgba(129,140,248,0.12);
    border-radius: 12px; padding: 36px 40px;
    display: flex; align-items: center; justify-content: space-between; gap: 32px;
}
.bottom-banner-text {}
.bottom-banner-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.6rem; letter-spacing: 2px; color: #fff; margin-bottom: 6px;
}
.bottom-banner-sub { font-size: 0.82rem; color: #44445a; font-weight: 300; }
.banner-btn .stButton > button {
    background: linear-gradient(135deg, #818cf8, #c084fc) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-size: 0.78rem !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    text-transform: uppercase !important; padding: 11px 24px !important;
    white-space: nowrap !important; width: auto !important; min-width: 0 !important;
    box-shadow: 0 4px 20px rgba(129,140,248,0.2) !important;
}

/* FOR YOU PAGE */
.foryou-section { padding: 40px 48px; }
.genre-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 32px; }
.rec-card {
    background: #0f1018; border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.05);
    padding: 14px 16px; margin-bottom: 10px;
    display: flex; gap: 14px; align-items: flex-start;
    transition: border-color 0.15s;
}
.rec-card:hover { border-color: rgba(129,140,248,0.25); }
.rec-num {
    font-family: 'Bebas Neue', cursive; font-size: 1.8rem;
    color: rgba(129,140,248,0.15); min-width: 36px; line-height: 1; text-align: right;
}
.rec-poster { width: 48px; height: 72px; border-radius: 6px; object-fit: cover; flex-shrink: 0; background: #1a1830; }
.rec-body { flex: 1; }
.rec-title { font-size: 0.88rem; font-weight: 600; color: #e0e0e8; margin-bottom: 5px; }
.rec-pills { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.rec-pill {
    background: rgba(129,140,248,0.08); color: #818cf8;
    font-size: 0.62rem; padding: 2px 8px; border-radius: 4px;
    border: 1px solid rgba(129,140,248,0.15);
}
.rec-bar-bg { height: 3px; background: rgba(255,255,255,0.05); border-radius: 2px; }
.rec-bar { height: 3px; background: linear-gradient(90deg, #818cf8, #c084fc); border-radius: 2px; }
.rec-score { font-size: 0.7rem; color: #818cf8; margin-top: 4px; }
.rec-view-btn .stButton > button {
    background: transparent !important; color: #818cf8 !important;
    border: 1px solid rgba(129,140,248,0.2) !important;
    border-radius: 6px !important; font-size: 0.68rem !important;
    font-weight: 500 !important; letter-spacing: 0.5px !important;
    text-transform: none !important; padding: 4px 12px !important;
    box-shadow: none !important; width: auto !important; min-width: 0 !important;
    margin-top: 6px !important;
}

/* DETAIL PAGE */
.detail-backdrop {
    width: 100%; height: 220px; position: relative; overflow: hidden;
}
.detail-backdrop img {
    width: 100%; height: 100%; object-fit: cover; opacity: 0.25; display: block;
}
.detail-backdrop-fade {
    position: absolute; bottom: 0; left: 0; right: 0; height: 120px;
    background: linear-gradient(to top, #07080f, transparent);
}
.detail-wrap { padding: 0 48px 48px; }
.detail-inner { display: flex; gap: 36px; align-items: flex-start; margin-top: -60px; position: relative; z-index: 2; }
.detail-poster {
    width: 180px; min-width: 180px; border-radius: 10px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.7); display: block;
}
.detail-poster-ph {
    width: 180px; min-width: 180px; height: 270px; border-radius: 10px;
    background: #1a1830; display: flex; align-items: center; justify-content: center; font-size: 3rem;
}
.detail-info { flex: 1; padding-top: 68px; }
.detail-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 2.6rem; letter-spacing: 2px; color: #fff;
    margin-bottom: 10px; line-height: 1;
}
.detail-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
.detail-rating {
    background: rgba(129,140,248,0.12); color: #818cf8;
    font-size: 0.75rem; font-weight: 600; padding: 3px 10px;
    border-radius: 5px; border: 1px solid rgba(129,140,248,0.2);
}
.detail-year, .detail-runtime { font-size: 0.78rem; color: #44445a; }
.detail-tagline { font-style: italic; color: #3d3d50; font-size: 0.83rem; margin-bottom: 14px; }
.detail-overview { font-size: 0.86rem; color: #8888a0; line-height: 1.75; max-width: 600px; margin-bottom: 18px; }
.detail-pill {
    display: inline-block; background: rgba(255,255,255,0.04);
    color: #aaa; font-size: 0.7rem; padding: 3px 12px;
    border-radius: 5px; border: 1px solid rgba(255,255,255,0.08);
    margin-right: 6px; margin-bottom: 6px;
}
.detail-sub-section { padding: 32px 48px; border-top: 1px solid rgba(255,255,255,0.04); }
.detail-sub-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.2rem; letter-spacing: 2px; color: #fff; margin-bottom: 18px;
}
.providers { display: flex; flex-wrap: wrap; gap: 12px; }
.provider {
    background: #0f1018; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);
    padding: 12px 16px; display: flex; flex-direction: column;
    align-items: center; gap: 6px; min-width: 88px;
}
.provider img { width: 42px; height: 42px; border-radius: 7px; object-fit: cover; }
.provider-name { font-size: 0.65rem; color: #666; text-align: center; }
.provider-type { font-size: 0.58rem; color: #44445a; }
.cast-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
    gap: 10px;
}
.cast-card { background: #0f1018; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.04); }
.cast-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; background: #1a1830; }
.cast-ph { width: 100%; aspect-ratio: 2/3; background: #1a1830; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; }
.cast-name { font-size: 0.68rem; font-weight: 600; color: #bbb; padding: 6px 7px 2px; }
.cast-char { font-size: 0.6rem; color: #3d3d50; padding: 0 7px 8px; }
.trailer-link {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(239,68,68,0.1); color: #f87171;
    font-size: 0.76rem; font-weight: 600; letter-spacing: 0.5px;
    padding: 9px 18px; border-radius: 7px; border: 1px solid rgba(239,68,68,0.2);
    text-decoration: none; margin-bottom: 0;
}
.back-btn .stButton > button {
    background: transparent !important; color: #666 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 7px !important; font-size: 0.74rem !important;
    font-weight: 500 !important; text-transform: none !important;
    letter-spacing: 0 !important; padding: 7px 16px !important;
    box-shadow: none !important; width: auto !important; min-width: 0 !important;
}

/* GLOBAL BUTTON FALLBACK */
.stButton > button {
    background: linear-gradient(135deg, #818cf8, #c084fc) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-size: 0.78rem !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    text-transform: uppercase !important; padding: 10px 22px !important;
    box-shadow: 0 4px 16px rgba(129,140,248,0.2) !important;
    transition: opacity 0.15s !important;
    width: auto !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: none !important; }

/* INPUTS */
.stTextInput input {
    background: #0f1018 !important; color: #e0e0e8 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; font-size: 0.85rem !important;
}
div[data-baseweb="select"] > div {
    background: #0f1018 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────
TMDB_KEY = "e27947a52a97677530fdd9e476e8b17c"
IMG_BASE = "https://image.tmdb.org/t/p"

GENRE_ICON = {
    'Action':'💥','Adventure':'🗺️','Animation':'🎨','Comedy':'😂',
    'Crime':'🕵️','Documentary':'📽️','Drama':'🎭','Fantasy':'🧙',
    'Horror':'👻','Musical':'🎵','Mystery':'🔍','Romance':'❤️',
    'Sci-Fi':'👽','Thriller':'🔪','War':'⚔️','Western':'🤠','Children':'🧒',
}

@st.cache_data(show_spinner=False)
def tmdb_search(title):
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    try:
        r = requests.get(f"https://api.themoviedb.org/3/search/movie",
                         params={"api_key": TMDB_KEY, "query": clean}, timeout=4)
        res = r.json().get('results', [])
        return res[0] if res else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def tmdb_details(title):
    result = tmdb_search(title)
    if not result:
        return None
    mid = result['id']
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{mid}",
                         params={"api_key": TMDB_KEY,
                                 "append_to_response": "credits,watch/providers,videos"},
                         timeout=5)
        return r.json()
    except Exception:
        return None

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

def genre_recs(picked, movies_df, n=12):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf.fit(movies_df['genres'])
    q = tfidf.transform(['|'.join(picked)])
    m = tfidf.transform(movies_df['genres'])
    s = cosine_similarity(q, m).flatten()
    idx = s.argsort()[::-1][:n]
    return pd.DataFrame([{
        'title':  movies_df.iloc[i]['title'],
        'genres': movies_df.iloc[i]['genres'],
        'score':  round(float(s[i]), 4)
    } for i in idx])

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    'page': 'home',
    'prev': 'home',
    'movie': None,
    'genres': [],
    'recs': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ratings_df, movies_df = load_data()

# ─────────────────────────────────────────────
# NAV BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="nav">
    <div class="nav-logo">MovieRecSys</div>
    <div class="nav-divider"></div>
</div>
""", unsafe_allow_html=True)

c1, c2, _sp = st.columns([1, 1, 16])
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
    # Back button
    st.markdown('<div style="padding:16px 48px 0;">', unsafe_allow_html=True)
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

    # Data
    ptitle   = det.get('title', title)
    overview = det.get('overview', '')
    tagline  = det.get('tagline', '')
    year     = det.get('release_date', '')[:4]
    rating   = round(det.get('vote_average', 0), 1)
    rt       = det.get('runtime') or 0
    runtime  = f"{rt//60}h {rt%60}m" if rt else ''
    genres   = [g['name'] for g in det.get('genres', [])]
    poster   = f"{IMG_BASE}/w400{det['poster_path']}"   if det.get('poster_path')   else None
    backdrop = f"{IMG_BASE}/w1280{det['backdrop_path']}" if det.get('backdrop_path') else None

    # Backdrop
    if backdrop:
        st.markdown(f"""
        <div class="detail-backdrop">
            <img src="{backdrop}" alt="backdrop"/>
            <div class="detail-backdrop-fade"></div>
        </div>""", unsafe_allow_html=True)

    # Hero row
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

    # Trailer
    videos  = det.get('videos', {}).get('results', [])
    trailer = next((v for v in videos if v.get('type') == 'Trailer'
                    and v.get('site') == 'YouTube'), None)
    if trailer:
        st.markdown(f"""
        <div class="detail-sub-section" style="border-top:none; padding-top:0; padding-bottom:24px;">
            <a class="trailer-link"
               href="https://www.youtube.com/watch?v={trailer['key']}"
               target="_blank">▶ &nbsp; Watch Trailer on YouTube</a>
        </div>""", unsafe_allow_html=True)

    # Where to watch
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
                   else '<div style="width:42px;height:42px;background:#1a1830;border-radius:7px;"></div>'
            cards += f"""<div class="provider">
                {img}
                <div class="provider-name">{p['provider_name']}</div>
                <div class="provider-type">{label}</div>
            </div>"""
        st.markdown(f'<div class="providers">{cards}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#3d3d50;font-size:0.82rem;">No streaming data available.</p>',
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
                <div class="cast-char">{c.get('character','')[:20]}</div>
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

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-label">AI-Powered Discovery</div>
        <div class="hero-title">Find <em>Cinema</em><br>You Will Love</div>
        <div class="hero-sub">
            Browse thousands of movies or let our AI recommend
            films tailored to your taste — no account needed.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cta-row">', unsafe_allow_html=True)
    if st.button("Get Personalised Recommendations", key="hero_cta"):
        st.session_state.page = 'recs'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

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
        f'<div style="font-size:0.68rem;color:#3d3d50;letter-spacing:1px;margin-bottom:20px;">'
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

        with cols[i % 8]:
            img_html = (f'<img class="mcard-img" src="{purl}" loading="lazy" alt="{short}"/>'
                        if purl else f'<div class="mcard-placeholder">{icon}</div>')
            st.markdown(f"""
            <div class="mcard">
                {img_html}
                <div class="mcard-hover">▶</div>
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

    # Bottom banner
    st.markdown('<div class="bottom-banner">', unsafe_allow_html=True)
    st.markdown("""
        <div class="bottom-banner-text">
            <div class="bottom-banner-title">Not sure what to watch?</div>
            <div class="bottom-banner-sub">
                Tell us your favourite genres and we will find the perfect movies for you.
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="banner-btn">', unsafe_allow_html=True)
    if st.button("Find My Movies", key="banner_cta"):
        st.session_state.page = 'recs'
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOR YOU PAGE
# ─────────────────────────────────────────────
elif st.session_state.page == 'recs':

    st.markdown("""
    <div class="hero" style="padding-bottom:48px;">
        <div class="hero-label">Personalised</div>
        <div class="hero-title" style="font-size:3rem;">Made <em>For You</em></div>
        <div class="hero-sub">Select your favourite genres and get instant recommendations.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="foryou-section">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-eyebrow" style="margin-bottom:14px;">Step 1 — Pick genres you love</div>',
        unsafe_allow_html=True
    )

    gcols  = st.columns(9)
    picked = []
    for idx, g in enumerate(all_genres):
        icon = GENRE_ICON.get(g, '🎬')
        with gcols[idx % 9]:
            if st.checkbox(f"{icon} {g}", key=f"gc_{g}",
                           value=g in st.session_state.genres):
                picked.append(g)
    st.session_state.genres = picked

    if picked:
        st.markdown(
            f'<div style="font-size:0.72rem;color:#818cf8;margin:16px 0 4px;'
            f'letter-spacing:1px;">SELECTED: {" · ".join(picked)}</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-eyebrow" style="margin-top:24px;margin-bottom:10px;">Step 2 — How many results?</div>',
        unsafe_allow_html=True
    )
    top_n = st.slider("", 5, 20, 10, label_visibility="collapsed", key="topn")

    b1, b2, _ = st.columns([1, 1, 8])
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

    if st.session_state.recs is not None:
        recs = st.session_state.recs
        st.markdown('<div class="divider" style="margin:32px 0;"></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Your Results</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-heading">Top {len(recs)} Picks For You</div>',
                    unsafe_allow_html=True)

        lc, rc = st.columns([1.1, 1])

        with lc:
            for i, row in recs.iterrows():
                bw    = int(row['score'] * 100)
                pills = ''.join(
                    f'<span class="rec-pill">{g.strip()}</span>'
                    for g in row['genres'].split('|') if g.strip()
                )
                pimg  = poster_url(row['title'])
                ph    = (f'<img class="rec-poster" src="{pimg}" alt="poster"/>'
                         if pimg else
                         f'<div class="rec-poster" style="display:flex;align-items:center;'
                         f'justify-content:center;font-size:1.4rem;">'
                         f'{GENRE_ICON.get(row["genres"].split("|")[0].strip(),"🎬")}</div>')

                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-num">0{i+1}</div>
                    {ph}
                    <div class="rec-body">
                        <div class="rec-title">{row['title']}</div>
                        <div class="rec-pills">{pills}</div>
                        <div class="rec-bar-bg">
                            <div class="rec-bar" style="width:{bw}%"></div>
                        </div>
                        <div class="rec-score">Match Score: {row['score']}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="rec-view-btn">', unsafe_allow_html=True)
                if st.button("View Details", key=f"rv_{i}"):
                    st.session_state.movie = row['title']
                    st.session_state.prev  = 'recs'
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        with rc:
            fig, ax = plt.subplots(figsize=(6, max(4, len(recs) * 0.5)))
            fig.patch.set_facecolor('#07080f')
            ax.set_facecolor('#0f1018')
            titles = [r['title'][:22] + '…' if len(r['title']) > 22
                      else r['title'] for _, r in recs.iterrows()]
            scores = recs['score'].values
            palette = ['#818cf8','#a78bfa','#c084fc','#e879f9','#f472b6',
                       '#818cf8','#a78bfa','#c084fc','#e879f9','#f472b6',
                       '#818cf8','#a78bfa','#c084fc','#e879f9','#f472b6',
                       '#818cf8','#a78bfa','#c084fc','#e879f9','#f472b6']
            bars = ax.barh(titles[::-1], scores[::-1],
                           color=palette[:len(scores)][::-1], height=0.54)
            for b in bars:
                ax.text(b.get_width() + 0.004,
                        b.get_y() + b.get_height() / 2,
                        f'{b.get_width():.2f}',
                        va='center', color='#44445a', fontsize=8)
            ax.set_xlabel('Match Score', color='#3d3d50', fontsize=8)
            ax.set_title('Your Match Chart', color='#aaa',
                         fontsize=10, fontweight='bold', pad=12)
            ax.tick_params(colors='#44445a', labelsize=7.5)
            for s in ax.spines.values():
                s.set_edgecolor('#1a1a2a')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown('<br>', unsafe_allow_html=True)
            st.download_button(
                "Download CSV",
                recs.to_csv(index=False),
                "recommendations.csv",
                "text/csv"
            )

    st.markdown('</div>', unsafe_allow_html=True)
