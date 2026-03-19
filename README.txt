# MovieRecSys

A personalized movie recommendation system built with Streamlit.

Supports two algorithms:
- Content-Based Filtering using TF-IDF on movie genres
- Collaborative Filtering using SVD (matrix factorization)

## Live Demo

https://movierecsys-euo4x4tzerekqnqtrcwnlt.streamlit.app

## Dataset

Uses the MovieLens dataset. Place the following files in a data/ folder:
- data/ratings.csv
- data/movies.csv

Download from: https://grouplens.org/datasets/movielens/

## Run Locally

pip install -r requirements.txt
streamlit run app.py

## Project Structure

MovieRecSys/
├── app.py
├── requirements.txt
└── data/
    ├── ratings.csv
    └── movies.csv
