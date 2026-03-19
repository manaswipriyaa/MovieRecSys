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
    page_title="MovieRecSys",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Righteous&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f0f4ff 0%, #fdf0ff 50%, #fff0f7 100%); color: #1a1a2e; }
    .title-wrap { text-align: center; padding: 2rem 0 0.5rem; }
    .title-main { font-family: 'Righteous', cursive; font-size: 3.2rem; background: linear-gradient(90deg, #6c63ff, #f64f59, #f9a825); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: 2px; }
    .title-sub { font-size: 0.9rem; color: #888; letter-spacing: 3px; text-transform: uppercase; margin-top: -6px; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #6c63ff 0%, #a855f7 100%) !important; }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] .stButton > button { background: white !important; color: #6c63ff !important; border: none; border-radius: 50px; font-family: 'Righteous', cursive; font-size: 1rem; letter-spacing: 1px; padding: 0.6rem 1.5rem; width: 100%; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: all 0.2s; }
    div[data-testid="stSidebar"] .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
    .stat-card { background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); border-radius: 12px; padding: 12px; text-align: center; }
    .stat-value { font-family: 'Righteous', cursive; font-size: 1.6rem; color: white; }
    .stat-label { font-size: 0.7rem; color: rgba(255,255,255,0.75); text-transform: uppercase; letter-spacing: 1px; }
    .movie-card { background: white; border-radius: 16px; padding: 20px; margin-bottom: 14px; box-shadow: 0 4px 20px rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.1); transition: transform 0.2s, box-shadow 0.2s; position: relative; overflow: hidden; }
    .movie-card::before { content: ''; position: absolute; top: 0; left: 0; width: 5px; height: 100%; background: linear-gradient(180deg, #6c63ff, #f64f59); border-radius: 16px 0 0 16px; }
    .movie-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(108,99,255,0.15); }
    .movie-rank { display: inline-block; background: linear-gradient(135deg, #6c63ff, #a855f7); color: white; font-family: 'Righteous', cursive; font-size: 0.75rem; padding: 3px 10px; border-radius: 50px; margin-bottom: 8px; }
    .movie-title { font-size: 1.05rem; font-weight: 500; color: #1a1a2e; margin-bottom: 5px; }
    .genre-pill { display: inline-block; background: #f0f4ff; color: #6c63ff; font-size: 0.7rem; padding: 2px 8px; border-radius: 50px; margin-right: 4px; margin-bottom: 4px; border: 1px solid rgba(108,99,255,0.2); }
    .score-bar-bg { background: #f0f0f0; border-radius: 50px; height: 6px; margin-top: 8px; }
    .score-bar { background: linear-gradient(90deg, #6c63ff, #f64f59); border-radius: 50px; height: 6px; }
    .score-text { font-size: 0.82rem; color: #6c63ff; font-weight: 500; margin-top: 5px; }
    .welcome-box { background: white; border-radius: 24px; padding: 60px 40px; text-align: center; box-shadow: 0 8px 40px rgba(108,99,255,0.1); margin-top: 40px; }
    .welcome-title { font-family: 'Righteous', cursive; font-size: 2rem; background: linear-gradient(90deg, #6c63ff, #f64f59); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 12px; }
    .welcome-sub { color: #888; font-size: 0.95rem; line-height: 1.7; }
    .section-header { font-family: 'Righteous', cursive; font-size: 1.4rem; color: #1a1a2e; margin-bottom: 4px; }
    .section-sub { font-size: 0.8rem; color: #aaa; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
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
    tfidf  = TfidfVectorizer(token_pattern=r'[^|]+')
    matrix = tfidf.fit_transform(movies_df['genres'])
    return tfidf, matrix

def get_content_recommendations(user_id, ratings_df, tfidf_matrix, movies_df, top_n=5):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if user_ratings.empty:
        return pd.DataFrame()
    user_movies  = pd.merge(user_ratings, movies_df, on='movieId')
    indices      = user_movies['movieId'].apply(lambda x: movies_df[movies_df['movieId'] == x].index[0])
    user_tfidf   = tfidf_matrix[indices]
    weights      = user_movies['rating'].values.reshape(-1, 1)
    user_profile = np.asarray(user_tfidf.multiply(weights).sum(axis=0)).reshape(1, -1)
    sims = cosine_similarity(user_profile, tfidf_matrix).flatten()
    seen = set(user_ratings['movieId'].tolist())
    results = []
    for i, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        row = movies_df.iloc[i]
        if row['movieId'] not in seen:
            results.append({'title': row['title'], 'genres': row['genres'], 'score': round(score, 4)})
        if len(results) >= top_n:
            break
    return pd.DataFrame(results)

def get_svd_recommendations(user_id, ratings_df, movies_df, top_n=5, sample_size=5000):
    try:
        sample    = ratings_df.sample(min(sample_size, len(ratings_df)), random_state=42)
        user_ids  = sample['userId'].unique()
        movie_ids = sample['movieId'].unique()
        u_idx = {u: i for i, u in enumerate(user_ids)}
        m_idx = {m: i for i, m in enumerate(movie_ids)}
        matrix = np.zeros((len(user_ids), len(movie_ids)))
        for _, row in sample.iterrows():
            if row['userId'] in u_idx and row['movieId'] in m_idx:
                matrix[u_idx[row['userId']], m_idx[row['movieId']]] = row['rating']
        row_means = np.true_divide(matrix.sum(1), (matrix != 0).sum(1).clip(min=1))
        for i in range(matrix.shape[0]):
            matrix[i][matrix[i] == 0] = row_means[i]
        U, sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
        k = min(50, len(sigma))
        reconstructed = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
        if user_id not in u_idx:
            return pd.DataFrame()
        uid  = u_idx[user_id]
        seen = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist())
        scores = [(m, reconstructed[uid, mi]) for m, mi in m_idx.items() if m not in seen]
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for m, score in scores[:top_n]:
            row = movies_df[movies_df['movieId'] == m]
            if not row.empty:
                results.append({'title': row.iloc[0]['title'], 'genres': row.iloc[0]['genres'], 'predicted_rating': round(float(score), 2)})
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()

# UI
st.markdown('<div class="title-wrap"><div class="title-main">MOVIERECSYS</div><div class="title-sub">Personalized Movie Recommendations</div></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

ratings_df, movies_df = load_data()

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### SETTINGS")
    st.markdown("---")
    if ratings_df is not None:
        user_ids = sorted(ratings_df['userId'].unique())
        user_id  = st.selectbox("User ID", user_ids, index=0)
        top_n    = st.slider("Recommendations", 3, 10, 5)
        method   = st.radio("Algorithm", ["Content-Based (TF-IDF)", "Collaborative (SVD)"])
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn  = st.button("GET RECOMMENDATIONS")
        st.markdown("---")
        st.markdown("### DATASET STATS")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{len(ratings_df):,}</div><div class="stat-label">Ratings</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{len(movies_df):,}</div><div class="stat-label">Movies</div></div>', unsafe_allow_html=True)
    else:
        st.error("Data files not found!")
        run_btn = False
        user_id = None

if ratings_df is None:
    st.markdown('<div class="welcome-box"><div class="welcome-title">Data Files Missing</div><p class="welcome-sub">Place your MovieLens dataset files in a data/ folder:<br>data/ratings.csv and data/movies.csv</p></div>', unsafe_allow_html=True)

elif 'run_btn' in dir() and run_btn:
    with st.spinner("Finding recommendations..."):
        if method == "Content-Based (TF-IDF)":
            _, tfidf_matrix = build_tfidf(movies_df)
            recs = get_content_recommendations(user_id, ratings_df, tfidf_matrix, movies_df, top_n)
            score_col, score_label = 'score', 'Similarity Score'
        else:
            recs = get_svd_recommendations(user_id, ratings_df, movies_df, top_n)
            score_col, score_label = 'predicted_rating', 'Predicted Rating'

    if recs.empty:
        st.warning(f"No recommendations found for User {user_id}.")
    else:
        col_cards, col_chart = st.columns([1, 1])
        with col_cards:
            st.markdown(f'<div class="section-header">Top {len(recs)} picks for User {user_id}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-sub">Algorithm: {method}</div>', unsafe_allow_html=True)
            for i, row in recs.iterrows():
                score     = row[score_col]
                bar_w     = int(min(score / 5.0, 1.0) * 100) if score_col == 'predicted_rating' else int(score * 100)
                genre_pills = ''.join([f'<span class="genre-pill">{g.strip()}</span>' for g in row['genres'].split('|') if g.strip()])
                st.markdown(f'<div class="movie-card"><div class="movie-rank">#{i+1}</div><div class="movie-title">{row["title"]}</div><div class="movie-genres">{genre_pills}</div><div class="score-bar-bg"><div class="score-bar" style="width:{bar_w}%"></div></div><div class="score-text">{score_label}: {score}</div></div>', unsafe_allow_html=True)

        with col_chart:
            st.markdown('<div class="section-header">Score Chart</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Visual comparison</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8f8ff')
            titles = [r['title'][:28] + ('...' if len(r['title']) > 28 else '') for _, r in recs.iterrows()]
            scores = recs[score_col].values
            bar_colors = ['#6c63ff', '#a855f7', '#f64f59', '#f9a825', '#00c9a7'][:len(scores)]
            bars = ax.barh(titles[::-1], scores[::-1], color=bar_colors[::-1], height=0.55)
            for bar in bars:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.3f}', va='center', color='#555', fontsize=9)
            ax.set_xlabel(score_label, color='#888', fontsize=9)
            ax.set_title(f'Top {len(recs)} Recommendations', color='#1a1a2e', fontsize=12, fontweight='bold', pad=12)
            ax.tick_params(colors='#666', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_edgecolor('#eee')
            ax.spines['bottom'].set_edgecolor('#eee')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("Download Recommendations CSV", recs.to_csv(index=False), "recommendations.csv", "text/csv")

else:
    st.markdown('<div class="welcome-box"><div class="welcome-title">Find Your Next Favourite Film</div><p class="welcome-sub">Select a User ID and algorithm from the sidebar,<br>then hit Get Recommendations.</p></div>', unsafe_allow_html=True)
