import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
from PIL import Image
import io
import base64


@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].fillna('')
    movies['keywords'] = movies['keywords'].fillna('')
    movies['combined'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords']
    return movies

movies = load_data()


@st.cache_resource
def compute_similarity(movies_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    vector_matrix = vectorizer.fit_transform(movies_df['combined'])
    similarity = cosine_similarity(vector_matrix, vector_matrix)
    return similarity

similarity = compute_similarity(movies)


def recommend(title, num_recommendations=5):
    if title.lower() not in movies['title'].str.lower().values:
        return []

    idx = movies[movies['title'].str.lower() == title.lower()].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for i in sorted_scores[1:num_recommendations+1]:
        movie_idx = i[0]
        movie_title = movies.iloc[movie_idx]['title']
        movie_id = movies.iloc[movie_idx]['id']
        recommendations.append({
            'title': movie_title,
            'id': movie_id
        })
    return recommendations


st.set_page_config(
    page_title="Movie Recommendations", 
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        color: white;
    }
    
    .stSelectbox > div > div > select {
        background-color: #333;
        color: white;
        border: 1px solid #555;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
    }
    
    .stSlider > div > div > div > div > div > div {
        background-color: #e50914;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #e50914, #b20710);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.4);
    }
    
    .movie-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #333;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
        border-color: #e50914;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        padding: 60px 0;
        text-align: center;
        border-bottom: 3px solid #e50914;
    }
    
    .app-logo {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(45deg, #e50914, #b20710);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    .hero-title {
        font-size: 36px;
        font-weight: 300;
        margin-bottom: 30px;
        color: #f5f5f5;
    }
    
    .hero-subtitle {
        font-size: 18px;
        color: #b3b3b3;
        margin-bottom: 40px;
    }
    
    .recommendations-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 30px;
    }
    
    .movie-title {
        font-size: 24px;
        font-weight: bold;
        color: #e50914;
        margin-bottom: 10px;
    }
    
    .movie-title a:hover {
        color: #ff6b6b !important;
        text-decoration: underline !important;
    }
    
    .movie-rank {
        background: linear-gradient(45deg, #e50914, #b20710);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 15px;
    }
    
    .controls-section {
        background: rgba(26, 26, 26, 0.8);
        padding: 30px;
        border-radius: 15px;
        margin: 30px 0;
        border: 1px solid #333;
    }
    
    .stMarkdown {
        color: white;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero-section">
    <div class="app-logo">Movie Recommendation</div>
    <div class="hero-title">Movie Recommendation Engine</div>
    <div class="hero-subtitle">Discover your next favorite movie with AI-powered suggestions</div>
</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="controls-section">
        <h2 style="text-align: center; color: #f5f5f5; margin-bottom: 30px;">🎯 Find Your Perfect Match</h2>
    </div>
    """, unsafe_allow_html=True)
    

    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "🎬 Select a movie you love:",
        sorted(movie_list),
        help="Choose a movie to get personalized recommendations"
    )
    

    num_recs = st.slider(
        "📊 Number of recommendations:",
        1, 10, 5,
        help="How many movie suggestions would you like?"
    )
    

    if st.button("🚀 Get Recommendations", use_container_width=True):
        with st.spinner("🎭 Analyzing your taste..."):
            recs = recommend(selected_movie, num_recs)
            
        if recs:
            st.markdown(f"""
            <div style="text-align: center; margin: 40px 0;">
                <h2 style="color: #e50914; margin-bottom: 20px;">🎉 Here's What You'll Love</h2>
                <p style="color: #b3b3b3; font-size: 18px;">Based on your love for <strong>{selected_movie}</strong></p>
                <p style="color: #b3b3b3; font-size: 14px; margin-top: 10px;">💡 Click on any movie title to view details on TMDB</p>
            </div>
            """, unsafe_allow_html=True)
            

            st.markdown('<div class="recommendations-grid">', unsafe_allow_html=True)
            
            for idx, movie in enumerate(recs, start=1):
                tmdb_url = f"https://www.themoviedb.org/movie/{movie['id']}"
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-rank">#{idx}</div>
                    <div class="movie-title">
                        <a href="{tmdb_url}" target="_blank" style="color: #e50914; text-decoration: none;">
                            {movie['title']} 🔗
                        </a>
                    </div>
                    <div style="color: #b3b3b3; font-size: 14px;">
                        🎬 Similar to your choice
                    </div>
                    <div style="margin-top: 10px;">
                        <a href="{tmdb_url}" target="_blank" style="color: #b3b3b3; font-size: 12px; text-decoration: underline;">
                            📖 View on TMDB
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            

            st.markdown("""
            <div style="text-align: center; margin: 40px 0; padding: 20px; background: rgba(229, 9, 20, 0.1); border-radius: 10px;">
                <p style="color: #b3b3b3; font-size: 16px;">
                    💡 <strong>Pro tip:</strong> Try different movies to discover diverse recommendations!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error("❌ Movie not found in our database. Please try another selection.")

st.markdown("""
<div style="text-align: center; margin-top: 60px; padding: 30px; border-top: 1px solid #333;">
    <p style="color: #666; font-size: 14px;">
        🎭 Powered by AI • Built with Streamlit • Movie data from TMDB
    </p>
</div>
""", unsafe_allow_html=True)
