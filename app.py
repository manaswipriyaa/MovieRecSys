import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

st.set_page_config(page_title="MovieRecSys", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #060714; color: #e8e8f0; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }
div[data-testid="stSidebar"] { display: none !important; }
footer { display: none !important; }
header { display: none !important; }

/* ── TOPBAR ── */
.topbar {
    position: sticky; top: 0; z-index: 100;
    background: linear-gradient(180deg, rgba(6,7,20,0.98) 0%, rgba(6,7,20,0.85) 100%);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0 48px;
    height: 64px;
    display: flex; align-items: center; justify-content: space-between;
}
.topbar-logo {
    font-family: 'Bebas Neue', cursive;
    font-size: 2rem;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #7c6dfa, #e040fb, #ff4081);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.topbar-nav { display: flex; gap: 32px; }
.topbar-navitem { font-size: 0.82rem; color: #888; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 500; }
.topbar-navitem.active { color: white; }

/* ── HERO BANNER ── */
.hero-banner {
    position: relative; width: 100%; height: 520px; overflow: hidden;
    background: linear-gradient(135deg, #0d0b2b 0%, #1a0533 50%, #0a1628 100%);
    display: flex; align-items: flex-end;
    padding: 0 64px 60px;
    margin-bottom: 0;
}
.hero-bg-glow {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 60% 60% at 70% 40%, rgba(124,109,250,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 30% 60%, rgba(224,64,251,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.hero-overlay {
    position: absolute; bottom: 0; left: 0; right: 0; height: 200px;
    background: linear-gradient(0deg, #060714 0%, transparent 100%);
}
.hero-content { position: relative; z-index: 2; max-width: 600px; }
.hero-badge {
    display: inline-block;
    background: rgba(124,109,250,0.2); color: #a78bfa;
    font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;
    padding: 5px 14px; border-radius: 50px;
    border: 1px solid rgba(124,109,250,0.35);
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 5rem; line-height: 0.95; letter-spacing: 3px;
    color: white; margin-bottom: 18px;
}
.hero-title span {
    background: linear-gradient(90deg, #7c6dfa, #e040fb);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-desc { font-size: 0.95rem; color: #9090b0; line-height: 1.7; margin-bottom: 28px; font-weight: 300; }
.hero-cta {
    display: inline-block;
    background: linear-gradient(135deg, #7c6dfa, #e040fb);
    color: white; font-size: 0.85rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 14px 36px; border-radius: 50px;
    box-shadow: 0 8px 32px rgba(124,109,250,0.35);
}

/* ── SECTION ── */
.section-wrap { padding: 40px 48px; }
.section-label {
    font-size: 0.68rem; letter-spacing: 3px; text-transform: uppercase;
    color: #7c6dfa; font-weight: 600; margin-bottom: 6px;
}
.section-heading {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.8rem; letter-spacing: 2px; color: white;
    margin-bottom: 24px;
}

/* ── GENRE PILLS ── */
.genre-pills { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 36px; }
.gpill {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    color: #aaa; font-size: 0.78rem; letter-spacing: 1px;
    padding: 7px 18px; border-radius: 50px;
    border: 1px solid rgba(255,255,255,0.09);
    cursor: pointer; transition: all 0.2s;
}
.gpill:hover { border-color: rgba(124,109,250,0.5); color: white; }
.gpill.active {
    background: linear-gradient(135deg, #7c6dfa22, #e040fb22);
    border-color: #7c6dfa; color: #c4b5fd;
}

/* ── MOVIE GRID ── */
.movie-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 14px; }
.mcard {
    border-radius: 10px; overflow: hidden;
    background: #10111f;
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
    cursor: pointer;
}
.mcard:hover {
    transform: scale(1.04) translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.6);
    border-color: rgba(124,109,250,0.4);
}
.mcard-poster {
    width: 100%; aspect-ratio: 2/3; object-fit: cover;
    display: block; background: #1a1b30;
}
.mcard-poster-placeholder {
    width: 100%; aspect-ratio: 2/3;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-size: 2.5rem; color: #333;
    background: linear-gradient(135deg, #12132a, #1e1040);
}
.mcard-info { padding: 10px 10px 12px; }
.mcard-title { font-size: 0.78rem; font-weight: 500; color: #ddd; line-height: 1.3; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mcard-genre { font-size: 0.65rem; color: #555; }

/* ── DIVIDER ── */
.divider { height: 1px; background: rgba(255,255,255,0.05); margin: 0 48px; }

/* ── FOR YOU / GENRE RECS ── */
.genre-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px; margin-bottom: 32px; }
.gcheckbox-card {
    background: #10111f; border-radius: 12px;
    padding: 18px 10px 14px; text-align: center;
    border: 2px solid rgba(255,255,255,0.06);
    transition: all 0.2s; cursor: pointer;
}
.gcheckbox-card:hover { border-color: rgba(124,109,250,0.35); }
.gcheckbox-card.selected { border-color: #7c6dfa; background: rgba(124,109,250,0.1); }
.gcheckbox-icon { font-size: 1.8rem; display: block; margin-bottom: 6px; }
.gcheckbox-name { font-size: 0.75rem; color: #bbb; letter-spacing: 0.5px; }

/* ── REC CARDS ── */
.rec-list { display: flex; flex-direction: column; gap: 14px; }
.rec-card {
    background: #10111f; border-radius: 14px;
    padding: 18px 20px; border: 1px solid rgba(255,255,255,0.06);
    display: flex; gap: 16px; align-items: flex-start;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: rgba(124,109,250,0.25); }
.rec-num {
    font-family: 'Bebas Neue', cursive;
    font-size: 2.2rem; line-height: 1;
    color: rgba(124,109,250,0.2); min-width: 44px; text-align: right;
}
.rec-poster { width: 54px; height: 80px; border-radius: 8px; object-fit: cover; background: #1a1b30; flex-shrink: 0; }
.rec-body { flex: 1; padding-top: 2px; }
.rec-title { font-size: 0.95rem; font-weight: 600; color: #eee; margin-bottom: 6px; }
.rec-pills { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }
.rec-pill {
    background: rgba(124,109,250,0.12); color: #a78bfa;
    font-size: 0.65rem; padding: 2px 8px; border-radius: 50px;
    border: 1px solid rgba(124,109,250,0.2);
}
.rec-bar-bg { background: rgba(255,255,255,0.07); border-radius: 50px; height: 4px; }
.rec-bar { background: linear-gradient(90deg, #7c6dfa, #e040fb); border-radius: 50px; height: 4px; }
.rec-score { font-size: 0.75rem; color: #7c6dfa; margin-top: 5px; font-weight: 500; }

/* Streamlit button override */
.stButton > button {
    background: linear-gradient(135deg, #7c6dfa, #e040fb) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important;
    font-family: 'Bebas Neue', cursive !important;
    font-size: 1rem !important; letter-spacing: 2px !important;
    padding: 0.65rem 2.5rem !important;
    box-shadow: 0 6px 24px rgba(124,109,250,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-2px) !important; }
.stSelectbox > div, .stTextInput > div > div > input {
    background: #10111f !important; color: #eee !important;
    border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── TMDB Poster Fetch ─────────────────────────────────────────────────────────
TMDB_KEY = "e27947a52a97677530fdd9e476e8b17c"
TMDB_IMG  = "https://image.tmdb.org/t/p/w300"

@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        import re
        clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_KEY, "query": clean},
            timeout=4
        )
        results = r.json().get('results', [])
        if results and results[0].get('poster_path'):
            return TMDB_IMG + results[0]['poster_path']
    except Exception:
        pass
    return None

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rp = os.path.join("data", "ratings.csv")
    mp = os.path.join("data", "movies.csv")
    if not os.path.exists(rp) or not os.path.exists(mp):
        return None, None
    ratings_df = pd.read_csv(rp)
    movies_df  = pd.read_csv(mp)
    movies_df['genres'] = movies_df['genres'].fillna('')
    return ratings_df, movies_df

@st.cache_data
def build_tfidf(movies_df):
    tfidf  = TfidfVectorizer(token_pattern=r'[^|]+')
    matrix = tfidf.fit_transform(movies_df['genres'])
    return tfidf, matrix

def get_genre_recommendations(selected_genres, movies_df, top_n=12):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf.fit(movies_df['genres'])
    query_vec  = tfidf.transform(['|'.join(selected_genres)])
    movie_vecs = tfidf.transform(movies_df['genres'])
    sims = cosine_similarity(query_vec, movie_vecs).flatten()
    top_idx = sims.argsort()[::-1][:top_n]
    results = []
    for i in top_idx:
        row = movies_df.iloc[i]
        results.append({'title': row['title'], 'genres': row['genres'], 'score': round(float(sims[i]), 4)})
    return pd.DataFrame(results)

GENRE_META = {
    'Action':'💥','Adventure':'🗺️','Animation':'🎨','Comedy':'😂',
    'Crime':'🕵️','Documentary':'📽️','Drama':'🎭','Fantasy':'🧙',
    'Horror':'👻','Musical':'🎵','Mystery':'🔍','Romance':'❤️',
    'Sci-Fi':'👽','Thriller':'🔪','War':'⚔️','Western':'🤠','Children':'🧒',
}

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [('page','home'), ('selected_genres',[]), ('genre_filter','All'), ('recs_ready', False), ('recs_df', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

ratings_df, movies_df = load_data()

# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">MOVIERECSYS</div>
    <div class="topbar-nav">
        <span class="topbar-navitem {'active' if st.session_state.page=='home' else ''}">Browse</span>
        <span class="topbar-navitem {'active' if st.session_state.page=='recs' else ''}">For You</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Nav buttons (invisible — styled via CSS)
nc1, nc2, nc3 = st.columns([1, 1, 10])
with nc1:
    if st.button("Browse", key="nb1"):
        st.session_state.page = 'home'
        st.rerun()
with nc2:
    if st.button("For You", key="nb2"):
        st.session_state.page = 'recs'
        st.rerun()

if ratings_df is None:
    st.error("Data files not found! Please add data/ratings.csv and data/movies.csv")
    st.stop()

all_genres = sorted({
    g.strip() for gs in movies_df['genres']
    for g in gs.split('|') if g.strip() and g.strip() != '(no genres listed)'
})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':

    # HERO
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-bg-glow"></div>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-badge">Powered by AI</div>
            <div class="hero-title">Discover<br><span>Cinema</span><br>You'll Love</div>
            <div class="hero-desc">Browse thousands of movies or let our AI recommend films based on your taste — no account needed.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero CTA button
    hc1, hc2, hc3 = st.columns([2, 1, 5])
    with hc1:
        if st.button("Get Personalised Recs", key="hero_cta"):
            st.session_state.page = 'recs'
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Browse section
    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Explore</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Browse Movies</div>', unsafe_allow_html=True)

    # Search + filter
    scol1, scol2 = st.columns([3, 1])
    with scol1:
        search = st.text_input("", placeholder="Search a movie title...", key="search_input", label_visibility="collapsed")
    with scol2:
        genre_pick = st.selectbox("Genre", ['All'] + all_genres, key="genre_pick", label_visibility="collapsed")

    # Filter
    if search:
        filtered = movies_df[movies_df['title'].str.contains(search, case=False, na=False)].head(40)
    elif genre_pick != 'All':
        filtered = movies_df[movies_df['genres'].str.contains(genre_pick, case=False, na=False)].sample(min(40, len(movies_df[movies_df['genres'].str.contains(genre_pick, case=False, na=False)])), random_state=1)
    else:
        filtered = movies_df.sample(min(40, len(movies_df)), random_state=42)

    st.markdown(f'<div style="font-size:0.78rem;color:#555;margin-bottom:20px;letter-spacing:1px;">{len(filtered)} TITLES</div>', unsafe_allow_html=True)

    # Movie grid — 8 per row
    cols = st.columns(8)
    for idx, (_, row) in enumerate(filtered.head(40).iterrows()):
        short = row['title'][:18] + ('…' if len(row['title']) > 18 else '')
        genre1 = row['genres'].split('|')[0].strip() if row['genres'] else ''
        poster_url = fetch_poster(row['title'])
        with cols[idx % 8]:
            if poster_url:
                st.markdown(f"""
                <div class="mcard">
                    <img class="mcard-poster" src="{poster_url}" alt="{short}" loading="lazy"/>
                    <div class="mcard-info">
                        <div class="mcard-title">{short}</div>
                        <div class="mcard-genre">{genre1}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                icon = GENRE_META.get(genre1, '🎬')
                st.markdown(f"""
                <div class="mcard">
                    <div class="mcard-poster-placeholder">{icon}</div>
                    <div class="mcard-info">
                        <div class="mcard-title">{short}</div>
                        <div class="mcard-genre">{genre1}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Bottom CTA strip
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d0b2b,#1a0533);padding:60px 48px;text-align:center;margin-top:40px;">
        <div style="font-family:'Bebas Neue',cursive;font-size:2.4rem;letter-spacing:3px;color:white;margin-bottom:12px;">
            Not sure what to watch?
        </div>
        <div style="color:#666;font-size:0.9rem;margin-bottom:28px;font-weight:300;">
            Tell us your favourite genres and we'll pick the perfect movies for you.
        </div>
    </div>
    """, unsafe_allow_html=True)
    bc1, bc2, bc3 = st.columns([3, 1, 3])
    with bc2:
        if st.button("Find My Movies", key="bottom_cta"):
            st.session_state.page = 'recs'
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FOR YOU
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'recs':

    st.markdown("""
    <div class="hero-banner" style="height:320px;">
        <div class="hero-bg-glow"></div>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-badge">Personalised</div>
            <div class="hero-title" style="font-size:3.5rem;">Made<br><span>For You</span></div>
            <div class="hero-desc">Select your favourite genres below and get instant recommendations.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">What do you like to watch?</div>', unsafe_allow_html=True)

    # Genre checkboxes
    gcols = st.columns(9)
    picked = []
    for i, genre in enumerate(all_genres):
        icon = GENRE_META.get(genre, '🎬')
        with gcols[i % 9]:
            val = st.checkbox(f"{icon}", key=f"g_{genre}",
                              value=(genre in st.session_state.selected_genres),
                              help=genre)
            if val:
                picked.append(genre)
            st.markdown(f'<div style="font-size:0.65rem;color:#888;text-align:center;margin-top:-8px;">{genre}</div>', unsafe_allow_html=True)
    st.session_state.selected_genres = picked

    if picked:
        st.markdown(f'<div style="color:#a78bfa;font-size:0.8rem;margin:16px 0 8px;letter-spacing:1px;">SELECTED: {" · ".join(picked)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:24px;">Step 2</div>', unsafe_allow_html=True)
    top_n = st.slider("How many recommendations?", 5, 20, 10, label_visibility="visible")

    rc1, rc2, rc3 = st.columns([1, 1, 4])
    with rc1:
        find_btn = st.button("Find My Movies", key="find_btn")
    with rc2:
        if st.button("Browse All", key="back_browse"):
            st.session_state.page = 'home'
            st.rerun()

    if find_btn:
        if not picked:
            st.warning("Please select at least one genre to continue.")
        else:
            with st.spinner("Finding your perfect movies..."):
                recs = get_genre_recommendations(picked, movies_df, top_n)
            st.session_state.recs_df    = recs
            st.session_state.recs_ready = True

    if st.session_state.recs_ready and st.session_state.recs_df is not None:
        recs = st.session_state.recs_df
        st.markdown('<div class="divider" style="margin:32px 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Your Results</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-heading">Top {len(recs)} Picks For You</div>', unsafe_allow_html=True)

        lcol, rcol = st.columns([1.1, 1])

        with lcol:
            for i, row in recs.iterrows():
                bar_w  = int(row['score'] * 100)
                pills  = ''.join([f'<span class="rec-pill">{g.strip()}</span>' for g in row['genres'].split('|') if g.strip()])
                poster = fetch_poster(row['title'])
                poster_html = f'<img class="rec-poster" src="{poster}" alt="poster"/>' if poster else f'<div class="rec-poster" style="display:flex;align-items:center;justify-content:center;font-size:1.5rem;background:#1a1b30;">{GENRE_META.get(row["genres"].split("|")[0].strip(),"🎬")}</div>'
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-num">0{i+1}</div>
                    {poster_html}
                    <div class="rec-body">
                        <div class="rec-title">{row['title']}</div>
                        <div class="rec-pills">{pills}</div>
                        <div class="rec-bar-bg"><div class="rec-bar" style="width:{bar_w}%"></div></div>
                        <div class="rec-score">Match Score: {row['score']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with rcol:
            fig, ax = plt.subplots(figsize=(6, max(5, len(recs) * 0.55)))
            fig.patch.set_facecolor('#060714')
            ax.set_facecolor('#10111f')
            titles = [r['title'][:22] + ('…' if len(r['title']) > 22 else '') for _, r in recs.iterrows()]
            scores = recs['score'].values
            palette = ['#7c6dfa','#a78bfa','#e040fb','#ff4081','#00e5ff',
                       '#69f0ae','#ffea00','#ff6d00','#7c6dfa','#a78bfa',
                       '#e040fb','#ff4081','#00e5ff','#69f0ae','#ffea00',
                       '#ff6d00','#7c6dfa','#a78bfa','#e040fb','#ff4081']
            bars = ax.barh(titles[::-1], scores[::-1], color=palette[:len(scores)][::-1], height=0.58)
            for bar in bars:
                ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.2f}', va='center', color='#888', fontsize=8)
            ax.set_xlabel('Match Score', color='#555', fontsize=8)
            ax.set_title('Your Movie Match Chart', color='#eee', fontsize=11, fontweight='bold', pad=14)
            ax.tick_params(colors='#666', labelsize=7.5)
            for s in ax.spines.values(): s.set_edgecolor('#222')
            ax.xaxis.label.set_color('#555')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("Download Recommendations CSV", recs.to_csv(index=False), "recommendations.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)
