# 📚 Goodreads Book Recommender & Search Engine  

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Goodreads_logo.svg" width="200">
</div>  

## 🚀 Overview  
This project builds a **book recommendation system** and a **search engine** using the **Goodreads dataset**.  
It applies **collaborative filtering** for personalized recommendations and **TF-IDF & Hashing Vectorization** for efficient book search.  

---

## 🔥 Features  
✅ **Book Search Engine**: Uses **`TF-IDF`** & **`HashingVectorizer`** with **`cosine similarity`** for relevant book searches.  
✅ **Personalized Recommendations**: Suggests books based on user preferences using **`collaborative filtering`**.  
✅ **Efficient Filtering**: Filters highly-rated books to enhance recommendation quality.  
✅ **Optimized for Performance**: Utilizes **`sparse matrices`**then transform to `compressed-Sparse-Row(CSR)` for memory-efficient computations.  
✅ **Interactive Display**: Shows **book titles, covers, ratings, and Goodreads links** for a rich user experience.  

---

## 📂 Dataset  
The project uses book metadata and user interactions from Goodreads.  
📥 **Download the dataset** [here](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/).  

### Dataset Files:  
- `goodreads_books.json.gz` → Book details  
- `goodreads_interactions.csv` → User interactions & ratings  
- `liked_books.csv` → User-preferred books  
- `book_id_map.csv` → Maps book IDs to internal indices  

---

## 🛠️ Technologies Used  
🔹 **Python** → Data Processing  
🔹 **Pandas** → Data Manipulation  
🔹 **Scikit-learn** → Vectorization & Similarity Calculation  
🔹 **NumPy** → Numerical Computation  
🔹 **SciPy** → Sparse Matrices for Efficient Storage  

---

## 🔍 Search Engine  
The search engine processes book titles, applies **TF-IDF & HashingVectorizer**, and retrieves similar books using **cosine similarity**.  

### 🔹 Implementation:  
```python
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_books(query, vectorizer, title_vectors, titles):
    query_vec = vectorizer.transform([query.lower()])
    similarity = cosine_similarity(query_vec, title_vectors).flatten()
    top_books = titles.iloc[np.argsort(-similarity)[:5]]
    return top_books
```
🤝 Recommendation System
This collaborative filtering approach finds users with similar book preferences and recommends books they haven't read yet.

- 🔹 How It Works:
- 1️⃣ Identify users with similar book preferences
- 2️⃣ Find books highly rated by those users
- 3️⃣ Recommend books that the target user hasn't read yet

🎯 Key Takeaways
- ✔️ Hybrid Approach → Combines content-based search & collaborative filtering
- ✔️ Efficient Computation → Uses sparse matrices for scalability
- ✔️ Scalable & Customizable → Adaptable for different recommendation scenarios
