import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="MovieRecSys", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Righteous&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0f0f1a; color: #f0f0f0; }

    /* NAV */
    .navbar { display: flex; align-items: center; justify-content: space-between; padding: 1rem 2rem; background: rgba(15,15,26,0.95); border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 2rem; }
    .nav-logo { font-family: 'Righteous', cursive; font-size: 1.6rem; background: linear-gradient(90deg, #6c63ff, #f64f59, #f9a825); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .nav-links { display: flex; gap: 24px; }
    .nav-link { color: #aaa; font-size: 0.9rem; cursor: pointer; letter-spacing: 1px; text-transform: uppercase; }
    .nav-link.active { color: white; font-weight: 500; }

    /* HERO */
    .hero { background: linear-gradient(135deg, #1a1040 0%, #0f0f1a 60%); border-radius: 20px; padding: 60px 50px; margin-bottom: 3rem; border: 1px solid rgba(108,99,255,0.2); position: relative; overflow: hidden; }
    .hero::before { content: ''; position: absolute; top: -60px; right: -60px; width: 300px; height: 300px; background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%); }
    .hero-tag { display: inline-block; background: rgba(108,99,255,0.2); color: #a78bfa; font-size: 0.75rem; padding: 4px 12px; border-radius: 50px; border: 1px solid rgba(108,99,255,0.3); margin-bottom: 16px; text-transform: uppercase; letter-spacing: 2px; }
    .hero-title { font-family: 'Righteous', cursive; font-size: 2.8rem; line-height: 1.1; margin-bottom: 12px; background: linear-gradient(90deg, #fff 60%, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .hero-sub { color: #888; font-size: 1rem; line-height: 1.6; max-width: 500px; }

    /* GENRE FILTER ROW */
    .genre-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 2rem; }
    .genre-chip { display: inline-block; background: rgba(255,255,255,0.06); color: #ccc; font-size: 0.8rem; padding: 6px 16px; border-radius: 50px; border: 1px solid rgba(255,255,255,0.1); cursor: pointer; transition: all 0.2s; }
    .genre-chip.selected { background: linear-gradient(135deg, #6c63ff, #a855f7); color: white; border-color: transparent; }

    /* SECTION HEADING */
    .section-title { font-family: 'Righteous', cursive; font-size: 1.3rem; color: white; margin-bottom: 1rem; padding-left: 4px; border-left: 4px solid #6c63ff; padding-left: 12px; }

    /* MOVIE CARD GRID */
    .movie-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 16px; margin-bottom: 3rem; }
    .mcard { background: #1a1a2e; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.06); transition: transform 0.2s, box-shadow 0.2s; cursor: pointer; }
    .mcard:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(108,99,255,0.2); border-color: rgba(108,99,255,0.3); }
    .mcard-poster { width: 100%; height: 110px; display: flex; align-items: center; justify-content: center; font-size: 2.5rem; }
    .mcard-body { padding: 10px 12px 12px; }
    .mcard-title { font-size: 0.82rem; font-weight: 500; color: #eee; margin-bottom: 4px; line-height: 1.3; }
    .mcard-genre { font-size: 0.68rem; color: #666; }

    /* REC CARDS */
    .rec-card { background: #1a1a2e; border-radius: 16px; padding: 20px; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.06); position: relative; overflow: hidden; display: flex; gap: 16px; align-items: flex-start; }
    .rec-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #6c63ff, #f64f59); }
    .rec-rank { font-family: 'Righteous', cursive; font-size: 2rem; color: rgba(108,99,255,0.3); min-width: 40px; }
    .rec-body { flex: 1; }
    .rec-title { font-size: 1rem; font-weight: 500; color: white; margin-bottom: 6px; }
    .rec-genres { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
    .rec-pill { display: inline-block; background: rgba(108,99,255,0.15); color: #a78bfa; font-size: 0.68rem; padding: 2px 8px; border-radius: 50px; border: 1px solid rgba(108,99,255,0.2); }
    .score-bar-bg { background: rgba(255,255,255,0.08); border-radius: 50px; height: 5px; }
    .score-bar { background: linear-gradient(90deg, #6c63ff, #f64f59); border-radius: 50px; height: 5px; }
    .score-text { font-size: 0.78rem; color: #6c63ff; margin-top: 5px; }

    /* GENRE SELECT CARDS */
    .genre-select-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; margin-bottom: 2rem; }
    .gcard { background: #1a1a2e; border-radius: 12px; padding: 16px 12px; text-align: center; border: 2px solid rgba(255,255,255,0.06); cursor: pointer; transition: all 0.2s; }
    .gcard:hover { border-color: rgba(108,99,255,0.4); }
    .gcard.picked { border-color: #6c63ff; background: rgba(108,99,255,0.12); }
    .gcard-icon { font-size: 1.8rem; margin-bottom: 6px; }
    .gcard-name { font-size: 0.8rem; color: #ccc; font-weight: 500; }

    /* Streamlit overrides */
    div[data-testid="stSidebar"] { display: none; }
    .stButton > button { background: linear-gradient(135deg, #6c63ff, #a855f7); color: white; border: none; border-radius: 50px; font-family: 'Righteous', cursive; font-size: 1rem; letter-spacing: 1px; padding: 0.6rem 2rem; transition: all 0.2s; }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
    .stTextInput input { background: #1a1a2e; color: white; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; }
    section[data-testid="stMain"] > div { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

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

def get_genre_recommendations(selected_genres, movies_df, tfidf_matrix, top_n=10):
    query = '|'.join(selected_genres)
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf.fit(movies_df['genres'])
    query_vec = tfidf.transform([query])
    movie_vecs = tfidf.transform(movies_df['genres'])
    sims = cosine_similarity(query_vec, movie_vecs).flatten()
    top_idx = sims.argsort()[::-1][:top_n]
    results = []
    for i in top_idx:
        row = movies_df.iloc[i]
        results.append({'title': row['title'], 'genres': row['genres'], 'score': round(sims[i], 4)})
    return pd.DataFrame(results)

# Genre emoji map
GENRE_META = {
    'Action':      {'icon': '💥', 'color': '#f64f59'},
    'Adventure':   {'icon': '🗺️', 'color': '#f9a825'},
    'Animation':   {'icon': '🎨', 'color': '#00c9a7'},
    'Comedy':      {'icon': '😂', 'color': '#f9a825'},
    'Crime':       {'icon': '🕵️', 'color': '#6c63ff'},
    'Documentary': {'icon': '📽️', 'color': '#888'},
    'Drama':       {'icon': '🎭', 'color': '#a855f7'},
    'Fantasy':     {'icon': '🧙', 'color': '#00c9a7'},
    'Horror':      {'icon': '👻', 'color': '#f64f59'},
    'Musical':     {'icon': '🎵', 'color': '#f9a825'},
    'Mystery':     {'icon': '🔍', 'color': '#6c63ff'},
    'Romance':     {'icon': '❤️', 'color': '#f64f59'},
    'Sci-Fi':      {'icon': '👽', 'color': '#6c63ff'},
    'Thriller':    {'icon': '🔪', 'color': '#a855f7'},
    'War':         {'icon': '⚔️', 'color': '#888'},
    'Western':     {'icon': '🤠', 'color': '#f9a825'},
    'Children':    {'icon': '🧒', 'color': '#00c9a7'},
}

def poster_bg(genres):
    for g in genres.split('|'):
        if g.strip() in GENRE_META:
            return GENRE_META[g.strip()]['color']
    return '#6c63ff'

def genre_icon(genres):
    for g in genres.split('|'):
        if g.strip() in GENRE_META:
            return GENRE_META[g.strip()]['icon']
    return '🎬'

# ── Session state ─────────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_genres' not in st.session_state:
    st.session_state.selected_genres = []
if 'browse_genre' not in st.session_state:
    st.session_state.browse_genre = 'All'

ratings_df, movies_df = load_data()

# ── NAVBAR ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="navbar">
    <div class="nav-logo">MOVIERECSYS</div>
    <div class="nav-links">
        <span class="nav-link {'active' if st.session_state.page == 'home' else ''}">Browse</span>
        <span class="nav-link {'active' if st.session_state.page == 'recs' else ''}">For You</span>
    </div>
</div>
""", unsafe_allow_html=True)

col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 6])
with col_nav1:
    if st.button("Browse", key="nav_home"):
        st.session_state.page = 'home'
        st.rerun()
with col_nav2:
    if st.button("For You", key="nav_recs"):
        st.session_state.page = 'recs'
        st.rerun()

if ratings_df is None:
    st.error("Data files missing! Add data/ratings.csv and data/movies.csv")
    st.stop()

all_genres = sorted(set(
    g.strip() for genres in movies_df['genres']
    for g in genres.split('|')
    if g.strip() and g.strip() != '(no genres listed)'
))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME — Browse Movies
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Discover</div>
        <div class="hero-title">Your next favourite<br>movie is here</div>
        <div class="hero-sub">Browse thousands of movies by genre, search for titles, or let us recommend based on your taste.</div>
    </div>
    """, unsafe_allow_html=True)

    # Search bar
    search = st.text_input("", placeholder="Search for a movie title...")

    # Genre filter
    genre_options = ['All'] + all_genres
    selected_browse = st.selectbox("Filter by genre", genre_options, index=0)

    # Filter movies
    if search:
        filtered = movies_df[movies_df['title'].str.contains(search, case=False, na=False)]
    elif selected_browse != 'All':
        filtered = movies_df[movies_df['genres'].str.contains(selected_browse, case=False, na=False)]
    else:
        filtered = movies_df.sample(min(60, len(movies_df)), random_state=42)

    st.markdown(f'<div class="section-title">{"Search Results" if search else selected_browse + " Movies"} <span style="color:#555;font-size:0.9rem;font-family:DM Sans;">({len(filtered)} found)</span></div>', unsafe_allow_html=True)

    # Movie grid — 5 per row
    cols = st.columns(5)
    for idx, (_, row) in enumerate(filtered.head(50).iterrows()):
        bg  = poster_bg(row['genres'])
        ico = genre_icon(row['genres'])
        short_title = row['title'][:22] + ('...' if len(row['title']) > 22 else '')
        genre_short = row['genres'].split('|')[0].strip() if row['genres'] else ''
        with cols[idx % 5]:
            st.markdown(f"""
            <div class="mcard">
                <div class="mcard-poster" style="background: linear-gradient(135deg, {bg}22, {bg}44);">
                    <span style="font-size:2.2rem">{ico}</span>
                </div>
                <div class="mcard-body">
                    <div class="mcard-title">{short_title}</div>
                    <div class="mcard-genre">{genre_short}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # CTA
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("Get Personalised Recs", key="goto_recs"):
            st.session_state.page = 'recs'
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FOR YOU — Genre-based Recommendations
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'recs':

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">For You</div>
        <div class="hero-title">Pick your favourite<br>genres</div>
        <div class="hero-sub">Select the genres you love and we'll recommend movies tailored just for you.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">What do you like to watch?</div>', unsafe_allow_html=True)

    # Genre selection grid using checkboxes styled as cards
    cols = st.columns(6)
    picked = []
    for i, genre in enumerate(all_genres):
        meta = GENRE_META.get(genre, {'icon': '🎬', 'color': '#6c63ff'})
        with cols[i % 6]:
            checked = st.checkbox(f"{meta['icon']} {genre}", key=f"genre_{genre}",
                                  value=(genre in st.session_state.selected_genres))
            if checked:
                picked.append(genre)

    st.session_state.selected_genres = picked

    st.markdown("<br>", unsafe_allow_html=True)

    if picked:
        st.markdown(f'<div style="color:#a78bfa;font-size:0.85rem;margin-bottom:1rem;">Selected: {", ".join(picked)}</div>', unsafe_allow_html=True)

    top_n = st.slider("Number of recommendations", 5, 20, 10)

    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        get_recs = st.button("Find My Movies", key="get_recs")

    if get_recs:
        if not picked:
            st.warning("Please select at least one genre.")
        else:
            with st.spinner("Finding your movies..."):
                _, tfidf_matrix = build_tfidf(movies_df)
                recs = get_genre_recommendations(picked, movies_df, tfidf_matrix, top_n)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">Your Recommendations — based on {", ".join(picked)}</div>', unsafe_allow_html=True)

            col_cards, col_chart = st.columns([1, 1])

            with col_cards:
                for i, row in recs.iterrows():
                    bar_w = int(row['score'] * 100)
                    pills = ''.join([f'<span class="rec-pill">{g.strip()}</span>' for g in row['genres'].split('|') if g.strip()])
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-rank">0{i+1}</div>
                        <div class="rec-body">
                            <div class="rec-title">{row['title']}</div>
                            <div class="rec-genres">{pills}</div>
                            <div class="score-bar-bg"><div class="score-bar" style="width:{bar_w}%"></div></div>
                            <div class="score-text">Match Score: {row['score']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_chart:
                fig, ax = plt.subplots(figsize=(7, max(4, len(recs) * 0.5)))
                fig.patch.set_facecolor('#0f0f1a')
                ax.set_facecolor('#1a1a2e')
                titles = [r['title'][:25] + ('...' if len(r['title']) > 25 else '') for _, r in recs.iterrows()]
                scores = recs['score'].values
                colors = ['#6c63ff','#a855f7','#f64f59','#f9a825','#00c9a7',
                          '#6c63ff','#a855f7','#f64f59','#f9a825','#00c9a7',
                          '#6c63ff','#a855f7','#f64f59','#f9a825','#00c9a7',
                          '#6c63ff','#a855f7','#f64f59','#f9a825','#00c9a7'][:len(scores)]
                bars = ax.barh(titles[::-1], scores[::-1], color=colors[::-1], height=0.6)
                for bar in bars:
                    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                            f'{bar.get_width():.3f}', va='center', color='#aaa', fontsize=8)
                ax.set_xlabel('Match Score', color='#888', fontsize=9)
                ax.set_title('Your Movie Matches', color='#fff', fontsize=12, fontweight='bold', pad=12)
                ax.tick_params(colors='#aaa', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("Download as CSV", recs.to_csv(index=False), "my_recommendations.csv", "text/csv")
