import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(
    page_title="🎬 MovieRecSys",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0d0d0d; }
    .stApp { background-color: #0d0d0d; color: #f0f0f0; }

    h1, h2, h3 {
        font-family: 'Bebas Neue', cursive;
        letter-spacing: 2px;
    }
    .title-text {
        font-family: 'Bebas Neue', cursive;
        font-size: 3.5rem;
        color: #e50914;
        letter-spacing: 4px;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle-text {
        text-align: center;
        color: #aaa;
        font-size: 0.95rem;
        margin-top: 0;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .movie-card {
        background: linear-gradient(135deg, #1a1a1a, #222);
        border: 1px solid #333;
        border-left: 4px solid #e50914;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: translateX(4px); border-left-color: #ff6b6b; }
    .movie-title { font-size: 1.1rem; font-weight: 600; color: #fff; margin-bottom: 4px; }
    .movie-genre { font-size: 0.8rem; color: #aaa; margin-bottom: 6px; }
    .rating-bar-bg { background: #333; border-radius: 10px; height: 6px; }
    .rating-bar { background: linear-gradient(90deg, #e50914, #ff6b6b); border-radius: 10px; height: 6px; }
    .rating-text { font-size: 0.85rem; color: #e50914; font-weight: 600; }
    .rank-badge {
        display: inline-block;
        background: #e50914;
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }
    .stButton>button {
        background: #e50914;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton>button:hover { background: #ff0f1f; }
    .stNumberInput input, .stSelectbox select {
        background: #1a1a1a;
        color: #f0f0f0;
        border: 1px solid #444;
    }
    .sidebar .sidebar-content { background-color: #111; }
    .metric-box {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e50914; }
    .metric-label { font-size: 0.75rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stSidebar"] { background-color: #111; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load ratings and movies CSVs from the data/ folder."""
    ratings_path = os.path.join("data", "ratings.csv")
    movies_path  = os.path.join("data", "movies.csv")

    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        return None, None

    ratings_df = pd.read_csv(ratings_path)
    movies_df  = pd.read_csv(movies_path)
    movies_df['genres'] = movies_df['genres'].fillna('')
    return ratings_df, movies_df

@st.cache_data
def build_tfidf(movies_df):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    matrix = tfidf.fit_transform(movies_df['genres'])
    return tfidf, matrix

# ── Recommendation Logic ──────────────────────────────────────────────────────

def get_content_recommendations(user_id, ratings_df, tfidf_matrix, movies_df, top_n=5):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if user_ratings.empty:
        return pd.DataFrame()

    user_movies = pd.merge(user_ratings, movies_df, on='movieId')
    indices = user_movies['movieId'].apply(
        lambda x: movies_df[movies_df['movieId'] == x].index[0]
    )
    user_tfidf  = tfidf_matrix[indices]
    weights     = user_movies['rating'].values.reshape(-1, 1)
    user_profile = np.asarray(user_tfidf.multiply(weights).sum(axis=0)).reshape(1, -1)

    sims = cosine_similarity(user_profile, tfidf_matrix).flatten()
    seen = set(user_ratings['movieId'].tolist())

    results = []
    for i, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        row = movies_df.iloc[i]
        if row['movieId'] not in seen:
            results.append({
                'title': row['title'],
                'genres': row['genres'],
                'score': round(score, 4)
            })
        if len(results) >= top_n:
            break

    return pd.DataFrame(results)

def get_svd_recommendations(user_id, ratings_df, movies_df, top_n=5, sample_size=10000):
    try:
        from surprise import Dataset, Reader, SVD  # type: ignore

        sample = ratings_df.sample(min(sample_size, len(ratings_df)), random_state=42)
        reader = Reader(rating_scale=(0.5, 5.0))
        data   = Dataset.load_from_df(sample[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        model  = SVD()
        model.fit(trainset)

        seen        = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist())
        all_movies  = sample['movieId'].unique()
        unseen      = [m for m in all_movies if m not in seen]
        preds       = [model.predict(user_id, m) for m in unseen]
        top_preds   = sorted(preds, key=lambda x: x.est, reverse=True)[:top_n]

        results = []
        for p in top_preds:
            row = movies_df[movies_df['movieId'] == int(p.iid)]
            if not row.empty:
                results.append({
                    'title': row.iloc[0]['title'],
                    'genres': row.iloc[0]['genres'],
                    'predicted_rating': round(p.est, 2)
                })
        return pd.DataFrame(results)
    except ImportError:
        return pd.DataFrame()

# ── Genre Emojis ──────────────────────────────────────────────────────────────

GENRE_EMOJI = {
    'Action':'🔥','Adventure':'🗺️','Comedy':'😂','Drama':'🎭',
    'Horror':'👻','Romance':'❤️','Sci-Fi':'👽','Animation':'🎨',
    'Fantasy':'🧙','Crime':'🕵️','Thriller':'🔪','Mystery':'🔍',
    'Documentary':'📽️','Musical':'🎵','Western':'🤠','War':'⚔️',
    'Children':'🧒','IMAX':'📺'
}

def genre_emojis(genres):
    return ' '.join(GENRE_EMOJI.get(g.strip(), '') for g in genres.split('|') if GENRE_EMOJI.get(g.strip()))

# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown('<div class="title-text">🎬 MOVIERECSYS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Personalized Movie Recommendations</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

ratings_df, movies_df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    if ratings_df is not None:
        user_ids = sorted(ratings_df['userId'].unique())
        user_id  = st.selectbox("Select User ID", user_ids, index=0)
        top_n    = st.slider("Number of Recommendations", 3, 10, 5)
        method   = st.radio("Algorithm", ["Content-Based (TF-IDF)", "Collaborative (SVD)"])
        run_btn  = st.button("🎬 Get Recommendations")

        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{len(ratings_df):,}</div><div class="metric-label">Ratings</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{len(movies_df):,}</div><div class="metric-label">Movies</div></div>', unsafe_allow_html=True)
    else:
        st.error("⚠️ Data files not found!\n\nPlease add:\n- `data/ratings.csv`\n- `data/movies.csv`")
        run_btn = False
        user_id = None

# Main content
if ratings_df is None:
    st.markdown("""
    <div style='background:#1a1a1a;border:1px solid #333;border-radius:10px;padding:30px;text-align:center;margin-top:40px;'>
        <h2 style='color:#e50914;font-family:Bebas Neue,cursive;letter-spacing:3px;'>DATA FILES MISSING</h2>
        <p style='color:#aaa;'>Place your MovieLens dataset files in a <code>data/</code> folder:</p>
        <pre style='background:#111;padding:14px;border-radius:6px;text-align:left;color:#ccc;display:inline-block;'>
MovieRecSys/
├── app.py
├── requirements.txt
└── data/
    ├── ratings.csv
    └── movies.csv</pre>
        <p style='color:#888;font-size:0.85rem;'>Download from: <a href='https://grouplens.org/datasets/movielens/' style='color:#e50914;'>grouplens.org/datasets/movielens</a></p>
    </div>
    """, unsafe_allow_html=True)

elif 'run_btn' in dir() and run_btn:
    with st.spinner("🎬 Finding your perfect movies..."):
        if method == "Content-Based (TF-IDF)":
            _, tfidf_matrix = build_tfidf(movies_df)
            recs = get_content_recommendations(user_id, ratings_df, tfidf_matrix, movies_df, top_n)
            score_col = 'score'
            score_label = 'Similarity Score'
        else:
            recs = get_svd_recommendations(user_id, ratings_df, movies_df, top_n)
            score_col = 'predicted_rating'
            score_label = 'Predicted Rating'

    if recs.empty:
        st.warning(f"No recommendations found for User {user_id}. Try a different user or algorithm.")
    else:
        st.markdown(f"### 🎯 Top {len(recs)} Picks for User #{user_id}")
        st.markdown(f"<p style='color:#888;font-size:0.85rem;'>Algorithm: {method}</p>", unsafe_allow_html=True)

        col_cards, col_chart = st.columns([1, 1])

        with col_cards:
            for i, row in recs.iterrows():
                emojis = genre_emojis(row['genres'])
                score  = row[score_col]
                bar_w  = int(min(score / 5.0, 1.0) * 100) if score_col == 'predicted_rating' else int(score * 100)

                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">
                        <span class="rank-badge">#{i+1}</span>{row['title']}
                    </div>
                    <div class="movie-genre">{emojis} {row['genres'].replace('|', ' · ')}</div>
                    <div class="rating-bar-bg"><div class="rating-bar" style="width:{bar_w}%"></div></div>
                    <div class="rating-text" style="margin-top:4px;">{score_label}: {score}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_chart:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#0d0d0d')
            ax.set_facecolor('#1a1a1a')

            titles  = [r['title'][:30] + ('…' if len(r['title']) > 30 else '') for _, r in recs.iterrows()]
            scores  = recs[score_col].values
            colors  = ['#e50914' if i == 0 else '#c0392b' if i == 1 else '#922b21' for i in range(len(scores))]

            bars = ax.barh(titles[::-1], scores[::-1], color=colors[::-1], height=0.6)
            for bar in bars:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.3f}', va='center', color='#ccc', fontsize=9)

            ax.set_xlabel(score_label, color='#aaa', fontsize=9)
            ax.set_title(f'Top {len(recs)} Recommendations', color='#fff', fontsize=12, fontweight='bold', pad=12)
            ax.tick_params(colors='#ccc', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            ax.xaxis.label.set_color('#aaa')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        csv = recs.to_csv(index=False)
        st.download_button("⬇️ Download Recommendations CSV", csv, "recommendations.csv", "text/csv")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
        <div style='font-size:5rem;margin-bottom:20px;'>🎬</div>
        <h2 style='font-family:Bebas Neue,cursive;color:#e50914;letter-spacing:3px;font-size:2.2rem;'>
            READY TO DISCOVER YOUR NEXT FAVOURITE FILM?
        </h2>
        <p style='color:#888;max-width:500px;margin:0 auto;line-height:1.7;'>
            Select a User ID and algorithm from the sidebar,<br>then hit <strong style='color:#e50914'>Get Recommendations</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
