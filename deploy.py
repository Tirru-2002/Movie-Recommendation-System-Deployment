import streamlit as st
import pickle
import pandas as pd

# Load your dataset
# Assuming 'ds' is your DataFrame containing movie information including 'Movie Name', 'Description'
ds = pd.read_csv(r'D:\DATA_SCIENCE_COURSE\PROJECT-1\movies_dataset(deploy).csv')
# Replace this with your actual DataFrame

similarity = pickle.load(open("similarity.pkl",'rb'))

# Streamlit app
st.title('Movie Recommendation System')

# User input for movie name
movie_name = st.selectbox('Select movie name:', ds['Movie Name'])

# Add background image using custom HTML and CSS
background_image = "https://img.freepik.com/free-photo/full-shot-scary-character-posing_23-2150701210.jpg?size=626&ext=jpg&ga=GA1.1.738268726.1702706112&semt=ais" # Replace with the URL of your background image
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image:linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.5)), url("{background_image}");
            background-size: cover;
            display: flex;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Recommendation function
def recommend_movies_by_name(movie_name):
    # Your existing recommendation function...
    # Check if the movie name exists in the DataFrame
    if movie_name in ds['Movie Name'].values:
        # Find the index of the movie with the given name
        movie_index = ds.index[ds['Movie Name'] == movie_name].tolist()[0]

        # Get the cosine similarity scores for the given movie index
        sim_scores = list(enumerate(similarity[movie_index]))

        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of recommended movies
        recommended_indices = [index for index, score in sim_scores[1:11]]  # Exclude the movie itself

        # Return recommended movies sorted by Movie Rating
        recommended_movies = ds.iloc[recommended_indices][['Movie Name','Year of Release','Description', 'Genre', 'Movie Rating']]
        recommended_movies = recommended_movies.sort_values(by='Movie Rating', ascending=False)

        return recommended_movies
    else:
        return f"Movie '{movie_name}' not found in the dataset."

# Display recommendations with styled table
recommendations = recommend_movies_by_name(movie_name)
if not isinstance(recommendations, str):
    st.subheader('Top 10 Recommended Movies:')
    # Apply custom CSS styles to the table
    st.markdown(
        """
        <style>
            table.dataframe {
                font-size: 18px;
                color: white;
                border-collapse: collapse;
                width: 100%;
                
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
                background-color: #000000;
            }
            th {
                background-color: #0e4b0e;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.table(recommendations[['Movie Name','Year of Release','Description', 'Genre']].reset_index(drop=True))
else:
    st.write(recommendations)
