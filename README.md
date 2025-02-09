# 📚 Goodreads Book Recommender & Search Engine  

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/1/1a/Goodreads_logo.svg" width="200">
</div>  

## 🚀 Overview  
This project builds a **`book recommendation system`** and a **`search engine`** using the **Goodreads dataset**.  
It applies **collaborative filtering** for personalized recommendations and **`TF-IDF` & `Hashing Vectorization`** for efficient book search.  

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
📥 **Download the dataset** [here](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/). 👈

### Dataset Files:  
- `goodreads_books.json.gz` → Book details  
- `goodreads_interactions.csv` → User interactions & ratings  
- `liked_books.csv` → User-preferred books  
- `book_id_map.csv` → Maps book IDs to internal indices
- `books_titles.json`-> to get metdata about books  

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

# Initialize vectorizers
vectorizer_hash = HashingVectorizer(n_features=2**20, alternate_sign=False)
vectorizer_tfidf = TfidfVectorizer()

# Precompute document vectors for both vectorizers
hash_vectors = vectorizer_hash.transform(book_deta['mod_title'])  # For HashingVectorizer
tfidf_vectors = vectorizer_tfidf.fit_transform(book_deta['mod_title'])  # For TfidfVectorizer


# style 
def enhance_url(val):
    return '<a target ="_blank" href="{}">GoodReads</a>'.format(val)
def enahce_imag(val):
    return '<a target="_blank"> <img src="{}" width =50></a>'.format(val)

# Unified search function
def search(query, vectorizer=vectorizer_tfidf, document_vectors=tfidf_vectors, titles=book_deta, top_k=5):
    """
    Search for books similar to the query using cosine similarity.

    Parameters:
        query (str): The search query.
        vectorizer: The vectorizer (HashingVectorizer or TfidfVectorizer).
        document_vectors: Precomputed vectors for the book titles.
        titles: DataFrame containing book titles and ratings.
        top_k (int): Number of top results to return.

    Returns:
        DataFrame: Top-k books sorted by ratings.
    """
    # Preprocess the query
    processed_query = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    
    # Vectorize the query
    query_vec = vectorizer.transform([processed_query])
    
    # Compute cosine similarity between the query and document vectors
    similarity = cosine_similarity(query_vec, document_vectors).flatten()
    
    # Get indices of top-k most similar books
    top_indices = np.argsort(similarity)[::-1][:top_k]
    
    # Retrieve top-k books and sort by ratings
    top_books = titles.iloc[top_indices].sort_values("ratings", ascending=False)
    
    return top_books.style.format({'url':enhance_url,'cover_image':enahce_imag})
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
