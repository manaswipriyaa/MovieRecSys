# 🎬 MovieRecSys

A personalized movie recommendation system built with Streamlit, supporting both Content-Based (TF-IDF) and Collaborative Filtering (SVD) algorithms.

## 🚀 Live Demo
> Deploy link will appear here after deployment

## 📁 Project Setup

### 1. Add the MovieLens Dataset

Download from [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/) and place files in a `data/` folder:

```
MovieRecSys/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── data/
    ├── ratings.csv
    └── movies.csv
```

### 2. Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app** → select this repo
5. Set **Main file path** to `app.py`
6. Click **Deploy!**

## 🎯 Features

- 🔍 Content-Based Filtering using TF-IDF on movie genres
- 🤖 Collaborative Filtering using SVD (matrix factorization)
- 🎨 Netflix-inspired dark UI
- 📊 Visual bar charts of recommendations
- ⬇️ Download recommendations as CSV
