# 🎬 AI Movie Recommendation Suite

> **Two recommendation engines. One movie discovery platform.**

An AI-powered movie recommendation project that explores two different approaches to personalized movie discovery:

* 🎯 **CineMind AI** — Hybrid recommendation using **content-based + collaborative filtering**
* 🧠 **Nexus** — Semantic movie discovery using **embeddings + Retrieval-Augmented Generation (RAG) + LLM reasoning**

Built with Python, Scikit-learn, Sentence Transformers, Hugging Face Transformers, and Gradio.

---

## ✨ What Makes This Project Different?

Most movie recommenders answer:

> **"If you liked this movie, what should you watch next?"**

This project explores a second question:

> **"I want a dark sci-fi thriller where astronauts discover something mysterious — what should I watch?"**

That creates two complementary recommendation experiences:

```text
                 🎬 MOVIE DISCOVERY
                        │
            ┌───────────┴───────────┐
            │                       │
       🎯 CINE MIND               🧠 NEXUS
            │                       │
     Movie-based                 Query-based
     recommendation              recommendation
            │                       │
   Content + Ratings         Semantic Search + LLM
```

---

# 🎯 1. CineMind AI

### Hybrid Movie Recommendation System

CineMind AI recommends movies by combining **movie content similarity** with **collaborative signals derived from user ratings**.

### 🔥 Features

* 🔎 Search for a movie
* 🎯 Hybrid recommendations
* 📝 Content-based filtering
* 👥 Collaborative filtering
* 🎭 Mood-based recommendations
* 🎲 Surprise Me mode
* 📊 Recommendation analytics
* ⭐ Rating and popularity signals
* 🎨 Interactive Gradio interface

---

## ⚙️ How CineMind Works

### Content-Based Filtering

Movie genres and tags are transformed into TF-IDF representations.

```text
Movie Metadata
      ↓
Genres + Tags
      ↓
TF-IDF
      ↓
Cosine Similarity
      ↓
Similar Movies
```

### Collaborative Filtering

User ratings are transformed into a user-movie matrix and reduced using **Truncated SVD**.

```text
User Ratings
      ↓
User × Movie Matrix
      ↓
Truncated SVD
      ↓
Latent Movie Representation
      ↓
Similarity
```

### Hybrid Score

The two signals are combined:

```text
Hybrid Score =
    Content Weight × Content Similarity
    +
    (1 - Content Weight) × Collaborative Similarity
```

This lets the user control the balance between **movie characteristics** and **rating behavior**.

---

# 🧠 2. Nexus

### Agentic RAG Movie Recommendation System

Nexus focuses on **natural-language movie discovery**.

Instead of selecting an existing movie, users describe what they want.

### Example

```text
"A sci-fi thriller about astronauts
discovering something mysterious in space."
```

Nexus converts the query into a semantic embedding, retrieves relevant movies, and uses an LLM to reason over the retrieved results.

---

## ⚙️ How Nexus Works

```text
Natural Language Query
          ↓
Sentence Transformer
          ↓
Query Embedding
          ↓
Semantic Retrieval
          ↓
Top-K Movie Candidates
          ↓
Movie Context
          ↓
Local LLM
          ↓
Recommendation + Explanation
```

### Core Technologies

* Sentence Transformers
* Semantic embeddings
* Vector similarity search
* Retrieval-Augmented Generation
* Local LLM inference
* Prompt-based reasoning

---

# 🆚 CineMind vs Nexus

| Capability                     | 🎯 CineMind AI | 🧠 Nexus |
| ------------------------------ | -------------- | -------- |
| Movie-to-movie recommendations | ✅              | ❌        |
| Natural-language search        | ❌              | ✅        |
| Content-based filtering        | ✅              | ✅        |
| Collaborative filtering        | ✅              | ❌        |
| TF-IDF                         | ✅              | ❌        |
| SVD                            | ✅              | ❌        |
| Semantic embeddings            | ❌              | ✅        |
| RAG                            | ❌              | ✅        |
| LLM reasoning                  | ❌              | ✅        |
| Mood recommendations           | ✅              | ❌        |
| Surprise Me                    | ✅              | ❌        |
| Analytics                      | ✅              | ❌        |
| AI-generated explanation       | ❌              | ✅        |
| Gradio UI                      | ✅              | ✅        |

---

# 🏗️ Repository Structure

```text
AI-Movie-Recommendation-Suite/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── CineMind_AI_Movie_Recommender.ipynb
│
├── src/
│   └── advanced_recommender.py
│
├── data/
│   └── README.md
│
└── screenshots/
    ├── cinemind.png
    └── nexus.png
```

> Dataset files are intentionally not included in the repository when they can be downloaded/generated during setup.

---

# 🛠️ Tech Stack

### Programming

`Python`

### Machine Learning

`Scikit-learn` · `Pandas` · `NumPy`

### Recommendation Systems

`TF-IDF` · `Cosine Similarity` · `Truncated SVD` · `Collaborative Filtering`

### NLP & Generative AI

`Sentence Transformers` · `Hugging Face Transformers` · `PyTorch` · `RAG` · `LLMs`

### UI

`Gradio` · `Matplotlib` · `Custom CSS`

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Movie-Recommendation-Suite.git

cd AI-Movie-Recommendation-Suite
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run CineMind AI

Open the notebook:

```text
notebooks/CineMind_AI_Movie_Recommender.ipynb
```

Run the cells sequentially.

The MovieLens dataset is downloaded and processed automatically.

---

# ▶️ Run Nexus

```bash
python src/advanced_recommender.py
```

The application launches a Gradio interface where you can enter natural-language movie descriptions.

---

# 💬 Example Searches

Try queries like:

```text
A sci-fi thriller where astronauts discover something unknown.
```

```text
A psychological thriller about someone questioning their memory.
```

```text
A futuristic movie about humans fighting an alien invasion.
```

```text
A mind-bending science fiction movie involving time travel.
```

```text
A movie about a giant ship hitting an iceberg.
```

---

# 📊 Dataset

### MovieLens

CineMind AI uses the **MovieLens** dataset for movie metadata and user ratings.

The MovieLens dataset provides:

* Movie titles
* Genres
* User ratings
* Rating counts

These are used to build the content-based and collaborative recommendation components.

---

# 🧪 Machine Learning Concepts

This project demonstrates practical implementation of:

* Content-Based Recommendation
* Collaborative Filtering
* Hybrid Recommendation
* TF-IDF Vectorization
* Cosine Similarity
* Matrix Factorization
* Truncated SVD
* Semantic Search
* Text Embeddings
* Retrieval-Augmented Generation
* LLM-based Reasoning
* Natural Language Recommendation

---

# 🎯 Project Goals

The project was designed to demonstrate how recommendation systems can evolve from traditional machine-learning approaches to modern AI-powered retrieval systems.

### Traditional ML

```text
Ratings + Metadata
       ↓
Similarity Models
       ↓
Recommendations
```

### Modern AI

```text
Natural Language
       ↓
Semantic Retrieval
       ↓
Relevant Context
       ↓
LLM Reasoning
       ↓
Recommendation
```

---

# 🚧 Future Improvements

* [ ] Add movie posters
* [ ] Add TMDB integration
* [ ] Add user profiles
* [ ] Add persistent user preferences
* [ ] Combine semantic + collaborative recommendations
* [ ] Add recommendation evaluation metrics
* [ ] Add Precision@K / Recall@K
* [ ] Add recommendation diversity scoring
* [ ] Add conversational recommendation history
* [ ] Add vector database
* [ ] Deploy the applications
* [ ] Add Docker support
* [ ] Add multilingual search

---

# 📌 Key Takeaway

This repository demonstrates **two fundamentally different recommendation paradigms**:

> 🎯 **CineMind AI** uses structured movie data and collaborative signals to recommend what you may like.

> 🧠 **Nexus** understands what you describe, retrieves semantically relevant movies, and uses an LLM to explain the recommendation.

Together, they provide a practical comparison between **classical recommender systems and modern Generative AI-powered recommendation**.

---

## ⭐ If You Like This Project

If you found this project useful or interesting, consider giving the repository a ⭐.

**Built with Python, Machine Learning, NLP, and Generative AI.**

🎬 **Discover smarter. Watch better.**
