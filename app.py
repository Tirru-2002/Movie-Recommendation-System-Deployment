import streamlit as st
import pickle
import pandas as pd
import requests

# Function to fetch poster URL from TMDb API
def get_poster_url(movie_id, api_key):
    base_url = "https://api.themoviedb.org/3/movie/"
    endpoint = f"{movie_id}?api_key={api_key}"
    response = requests.get(base_url + endpoint)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return None

# Load your dataset
# Assuming 'ds' is your DataFrame containing movie information including 'Movie Name', 'Description', and 'TMDb_ID'
# Replace this with your actual DataFrame
movies = pickle.load(open("movies_list.pkl",'rb'))
similarity = pickle.load(open("similarity.pkl",'rb'))
movies_list = movies['Movie Name'].values

# Streamlit app
st.title('Movie Recommendation System')

# User input for movie name
movie_name = st.selectbox('select movie name:', movies_list)

# Recommendation function
def recommend_movies_by_name(movie_name, ds=ds, api_key="c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"):
    # Your existing recommendation function...
    # Check if the movie name exists in the DataFrame
    if movie_name in ds['Movie Name'].values:
        # Find the index of the movie with the given name
        movie_index = ds.index[ds['Movie Name'] == movie_name].tolist()[0]

        # Get the cosine similarity scores for the given movie index
        sim_scores = list(enumerate(cosine_sim[movie_index]))

        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of recommended movies
        recommended_indices = [index for index, score in sim_scores[1:11]]  # Exclude the movie itself

        # Return recommended movies
        return ds.iloc[recommended_indices][['Movie Name', 'Description','Genre']]
    else:
        return f"Movie '{movie_name}' not found in the dataset."

# Display recommendations
recommendations = recommend_movies_by_name(movie_name)
if not isinstance(recommendations, str):
    st.subheader('Top 10 Recommended Movies:')
    for index, row in recommendations.iterrows():
        st.write(f"**{row['Movie Name']}**")
        st.write(row['Description'])
        
        # Fetch poster URL from TMDb API
        poster_url = get_poster_url(row['TMDb_ID'], api_key)
        if poster_url:
            st.image(poster_url, caption=row['Movie Name'], use_column_width=True)
        else:
            st.write("Poster not available")
        
        st.markdown("---")
else:
    st.write(recommendations)
