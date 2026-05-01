import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize



# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# -----------------------------------
# LOAD PICKLED OBJECTS (ONCE)
# -----------------------------------

@st.cache_resource
def load_model():
    import pickle
    import faiss

    index = faiss.read_index("faiss.index")

    with open("metadata.pkl", "rb") as f:
        data = pickle.load(f)

   

    return index, data["ds"], data["movie_to_idx"]

index, ds, movie_to_idx = load_model()

# -----------------------------------
# UI STYLING
# -----------------------------------
st.markdown(
    """
    <style>
        .title-text {
            font-size: 36px;
            color: #ffffff;
            text-align: center;
        }
        .subheader-text {
            font-size: 24px;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title-text">Movie Recommendation System</h1>', unsafe_allow_html=True)


# -----------------------------------
# USER-CONTROLLED WEIGHTS
# -----------------------------------
preset = st.sidebar.selectbox(
    "Preset Mode",
    ["Balanced", "Director Focus", "Star Focus", "Genre Focus"]
)

if preset == "Director Focus":
    semantic_w, director_w, star_w, genre_w, sequel_w = 0.4, 0.35, 0.15, 0.05, 0.05

elif preset == "Star Focus":
    semantic_w, director_w, star_w, genre_w, sequel_w = 0.4, 0.15, 0.30, 0.10, 0.05

elif preset == "Genre Focus":
    semantic_w, director_w, star_w, genre_w, sequel_w = 0.4, 0.10, 0.15, 0.30, 0.05

else:  # Balanced
    semantic_w, director_w, star_w, genre_w, sequel_w = 0.55, 0.20, 0.15, 0.06, 0.04



# -----------------------------------
# USER INPUT
# -----------------------------------
movie_name = st.selectbox("Select movie:", ds["Movie Name"].values)

st.sidebar.markdown("### 🔍 Current Weights")
st.sidebar.write({
    "Semantic": f"{round(semantic_w, 3)*100:.0f}%",
    "Director": f"{round(director_w, 3)*100:.0f}%",
    "Stars": f"{round(star_w, 3)*100:.0f}%",
    "Genre": f"{round(genre_w, 3)*100:.0f}%",
    "Sequel": f"{round(sequel_w, 3)*100:.0f}%"
})


# -----------------------------
# 1) Clean text helper
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text) # Remove punctuation, special chars
    text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace
    return text

# -----------------------------
# 2) Sequel / franchise heuristic
# -----------------------------
def sequel_root(title):
    title = clean_text(title)

    # Remove common sequel markers
    title = re.sub(r"\b(part|chapter|volume|vol|episode|ep)\s*\d+\b", "", title) # Remove part/chapter numbers
    title = re.sub(r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b", "", title) # Remove Roman numerals
    title = re.sub(r"\b2nd|3rd|4th\b", "", title) # Remove ordinal numbers
    title = re.sub(r"\s+", " ", title).strip() # Normalize whitespace
    return title

def sequel_bonus(title_a, title_b):
    root_a = sequel_root(title_a)
    root_b = sequel_root(title_b)
    if root_a and root_b and root_a == root_b:
        return 1.0
    if root_a and root_b and (root_a in root_b or root_b in root_a):
        return 0.7
    return 0.0

# -----------------------------------
# 3) HYBRID RECOMMENDATION FUNCTION
# -----------------------------------

def overlap_score(a, b):
    set_a = set(clean_text(a).split())
    set_b = set(clean_text(b).split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))

def recommend_movies(movie_name, top_n=10, candidate_k=50,semantic_w=0.55,
                     director_w=0.20,
                     star_w=0.15,
                     genre_w=0.06,
                     sequel_w=0.04):
    movie_name_clean = movie_name.lower().strip()

    if movie_name_clean not in movie_to_idx:
        raise ValueError(f"Movie '{movie_name}' not found in dataset.")

    query_idx = movie_to_idx[movie_name_clean]
    query_row = ds.iloc[query_idx]
    # query_vec = embeddings[query_idx].reshape(1, -1).astype(np.float32)
    query_vec = index.reconstruct(query_idx).reshape(1, -1).astype(np.float32)

    # Retrieve candidates from FAISS
    scores, indices = index.search(query_vec, candidate_k)

    query_title = query_row["Movie Name"]
    query_genre = query_row["Genre"]
    query_director = query_row["Director"]
    query_stars = query_row["Stars"]

    candidates = []

    for idx, semantic_sim in zip(indices[0], scores[0]):
        if idx == query_idx:
            continue

        row = ds.iloc[idx]

        director_match = 1.0 if clean_text(row["Director"]) == clean_text(query_director) and clean_text(query_director) else 0.0
        star_match = overlap_score(row["Stars"], query_stars)
        genre_match = overlap_score(row["Genre"], query_genre)
        sequel = sequel_bonus(query_title, row["Movie Name"])

        final_score = (
            semantic_w * float(semantic_sim) +
            director_w * director_match +
            star_w * star_match +
            genre_w * genre_match +
            sequel_w * sequel
        )

        candidates.append({
            "Movie Name": row["Movie Name"],
            "Genre": row["Genre"],
            "Director": row["Director"],
            "Stars": row["Stars"],
            "Description": row["Description"],
            # "semantic_similarity": float(semantic_sim),
            # "director_match": director_match,
            # "star_match": star_match,
            # "genre_match": genre_match,
            # "sequel_bonus": sequel,
            "final_score": final_score
        })


    result = pd.DataFrame(candidates).sort_values("final_score", ascending=False).head(top_n)
    return result.reset_index(drop=True)

# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------
results = recommend_movies(movie_name,semantic_w=semantic_w,
    director_w=director_w,
    star_w=star_w,
    genre_w=genre_w,
    sequel_w=sequel_w)




if results is not None:
    st.markdown('<h2 class="subheader-text">Top Recommendations</h2>', unsafe_allow_html=True)
    st.table(results.reset_index(drop=True))
else:
    st.warning("Movie not found")
