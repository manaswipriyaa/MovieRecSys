# MovieRecSys - Hybrid Movie Recommendation System

A college project that builds a hybrid movie recommendation engine combining collaborative filtering and content-based filtering - deployed as an interactive web app using Streamlit.

---

## Problem Statement

Streaming platforms recommend content based on user behaviour and movie attributes. Pure collaborative filtering suffers from the cold-start problem, while pure content-based filtering ignores community preferences. This project combines both approaches into a hybrid engine for more relevant, personalised recommendations.

---

## Dataset

- **Source:** MovieLens 100K Dataset (GroupLens Research)
- **Size:** 100,000 ratings from 943 users across 1,682 movies
- **Features used:** User ratings matrix, movie genres, titles, tags

---

## Approach

### 1. Collaborative Filtering (SVD)
- Used Singular Value Decomposition (SVD) via the `Surprise` library
- Learns latent user-movie interaction patterns from the ratings matrix
- Tuned with cross-validation to minimise RMSE

### 2. Content-Based Filtering
- Built TF-IDF vectors from movie metadata (genres, tags, descriptions)
- Computed cosine similarity between movies to find content neighbours

### 3. Hybrid Weighting
- Combined both scores using a weighted average
- Tuned the weight ratio to optimise recommendation relevance
- Achieved **12% RMSE improvement** over pure collaborative filtering baseline

---

## Results

| Model | RMSE |
|---|---|
| Collaborative Filtering (SVD) only | ~0.98 |
| **Hybrid (SVD + Content-Based)** | **~0.87** |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Recommendation | Surprise (SVD), Scikit-learn (TF-IDF, cosine similarity) |
| Data | Pandas |
| Web App | Streamlit, HTML |
| Notebook | Jupyter Notebook |

---

## Project Structure

```
MovieRecSys/
│
├── data/
│   ├── ratings.csv
│   └── movies.csv
├── notebooks/
│   └── recommendation_system.ipynb
├── app/
│   └── app.py              # Streamlit web app
├── outputs/
│   └── rmse_comparison.png
└── README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/manaswipriyaa/MovieRecSys.git
cd MovieRecSys

# Install dependencies
pip install pandas scikit-learn scikit-surprise streamlit jupyter

# Run the Streamlit app
streamlit run app/app.py

# Or explore the notebook
jupyter notebook notebooks/recommendation_system.ipynb
```

---

## Features of the Web App

- Enter a movie name and get top 10 personalised recommendations
- Recommendations include genre tags and streaming platform links
- Clean, interactive UI built with Streamlit

---

## Author

**Manaswi Priya Maddu**
B.Tech - AI & Machine Learning | Acharya Nagarjuna University
*(College Project - Acharya Nagarjuna University, 2025)*
[LinkedIn](https://linkedin.com/in/manaswi-priya-2126481b8) | [GitHub](https://github.com/manaswipriyaa)
