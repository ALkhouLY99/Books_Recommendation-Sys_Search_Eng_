# 📚 Recommendation System & Search Engine

<div align="center">
    <h1><strong>Goodreads Book Recommender & Search Engine</strong></h1>
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Goodreads_logo.svg" width="200">
</div>

## 🚀 Overview
This project builds a **book recommendation system** and a **search engine** using **Goodreads dataset**.  
It applies **collaborative filtering** for recommendations and **TF-IDF & Hashing Vectorization** for search functionality.  

## 🔥 Features
- **Book Search Engine**: Finds books using **TF-IDF** & **HashingVectorizer** with **cosine similarity**.
- **Personalized Recommendations**: Uses **collaborative filtering** to suggest books based on user preferences.
- **Data Processing**: Cleans and transforms book titles for better search and recommendation results.
- **Interactive Display**: Results include **book titles, covers, ratings, and Goodreads links**.

---

## 📂 Dataset
Download the **Goodreads dataset** from [here](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/).  
We use:
- `goodreads_books.json.gz`: Contains book details.
- `goodreads_interactions.csv`: User interactions and ratings.
- `liked_books.csv`: User-preferred books.
- `book_id_map.csv`: Maps book IDs to their internal indices.

---
📚 Recommendation System & Search Engine 🔍
<div align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Goodreads_logo.svg/2560px-Goodreads_logo.svg.png" width="250"> </div>
🚀 Project Overview
This project builds a book recommendation system and search engine using machine learning techniques to enhance book discovery. It processes Goodreads book data, applies text vectorization (TF-IDF & Hashing) for searching, and implements collaborative filtering to suggest books based on user preferences.

✨ Features
✔️ Book Search Engine using TF-IDF and HashingVectorizer
✔️ Personalized Book Recommendations via collaborative filtering
✔️ Efficient filtering of highly-rated books
✔️ Optimized for performance using sparse matrices

📂 Dataset
📌 The project uses Goodreads book interactions and metadata. You can download the dataset here:
📥 Goodreads Dataset (Replace with actual link)

🛠️ Technologies Used
🔹 Python - Data Processing
🔹 Pandas - Data Manipulation
🔹 Scikit-learn - Vectorization & Similarity Calculation
🔹 NumPy - Numerical Computation
🔹 Scipy - Sparse Matrices for Efficient Storage

🔍 Search Engine
The search engine cleans book titles, applies TF-IDF & HashingVectorizer, and retrieves similar books using cosine similarity.
``` pythhon
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_books(query, vectorizer, title_vectors, titles):
    query_vec = vectorizer.transform([query.lower()])
    similarity = cosine_similarity(query_vec, title_vectors).flatten()
    top_books = titles.iloc[np.argsort(-similarity)[:5]]
    return top_books
```
🤝 Recommendation System
The recommendation engine is based on collaborative filtering. It finds users with similar book preferences and suggests books they liked but the current user hasn’t read.

🔹 Steps:
1️⃣ Identify users who liked similar books
2️⃣ Find books rated highly by those users
3️⃣ Recommend books not yet read by the target user

```python
from sklearn.metrics.pairwise import cosine_similarity

def recommend_books(user_id, interactions, similarity_matrix):
    similar_users = np.argsort(-similarity_matrix[user_id])[:10]
    recommendations = interactions[interactions['user_id'].isin(similar_users)]
    return recommendations
```
🎯 Key Takeaways
✔️ Hybrid Approach combining content-based search & collaborative filtering
✔️ Efficient Sparse Matrix Computations to handle large datasets
✔️ Scalable and Customizable for different recommendation scenarios

💡 Developed with ❤️ by Abdo
🚀 Keep Reading, Keep Exploring!

