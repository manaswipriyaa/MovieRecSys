# 🎬 Movie Recommendation System

A **Hybrid Movie Recommender** that blends **Content-Based Filtering** and **Collaborative Filtering** to provide personalized suggestions. Users can explore movie recommendations along with visuals, metadata, trailers, and streaming platform links — all wrapped in an interactive UI with light/dark mode support.

## 🔥 Features

- 🎯 **Hybrid Recommendation Logic**  
  Uses **SVD** (collaborative) + **cosine similarity** (content-based) for improved predictions.
  
- 🖼️ **Dynamic Posters & Metadata**  
  Automatically fetches posters, genre badges, and ratings using **TMDB API**.

- 🎞️ **Watch Trailers**  
  Trailer previews embedded alongside recommendations.

- 📺 **Streaming Platform Links**  
  Stream directly on platforms like Netflix, Prime Video, etc., with clickable icons.

- 🌙 **Light/Dark Theme Toggle**  
  Toggle themes with smooth transitions and glowing UI effects.

- 💻 **Interactive UI**  
  Styled with modern HTML/CSS and enhanced using JavaScript.

---

## 🛠️ Tech Stack

| Layer       | Tools/Technologies                       |
|-------------|------------------------------------------|
| Frontend    | HTML5, CSS3, JavaScript (Fetch API)      |
| Backend     | Python, Flask                            |
| ML Models   | SVD (Surprise Lib), Cosine Similarity    |
| APIs Used   | TMDB API for posters, metadata, trailers |

---

## 📊 How It Works

1. **Collaborative Filtering** recommends based on user-item matrix using **SVD**.
2. **Content-Based Filtering** matches movie features like genre, keywords, and description using **cosine similarity**.
3. Both results are combined (weighted average) for the final recommendation list.
4. Posters, ratings, genres, trailers, and stream links are rendered on a visually appealing webpage.

---

## 📸 Sample Output

<img width="1891" height="825" alt="Screenshot 2025-07-27 231231" src="https://github.com/user-attachments/assets/03782e61-d42f-46ea-9002-1e747cacb94f" />

