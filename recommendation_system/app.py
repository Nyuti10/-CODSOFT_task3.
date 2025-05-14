import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
movies = [
    {'id': 1, 'title': 'The Shawshank Redemption', 'genres': 'Drama'},
    {'id': 2, 'title': 'The Godfather', 'genres': 'Crime, Drama'},
    {'id': 3, 'title': 'The Dark Knight', 'genres': 'Action, Crime, Drama'},
    {'id': 4, 'title': 'Pulp Fiction', 'genres': 'Crime, Drama'},
    {'id': 5, 'title': 'Forrest Gump', 'genres': 'Drama, Romance'},
    {'id': 6, 'title': 'Inception', 'genres': 'Action, Adventure, Sci-Fi'},
    {'id': 7, 'title': 'The Matrix', 'genres': 'Action, Sci-Fi'},
    {'id': 8, 'title': 'Goodfellas', 'genres': 'Crime, Drama'},
    {'id': 9, 'title': 'The Silence of the Lambs', 'genres': 'Crime, Drama, Thriller'},
    {'id': 10, 'title': 'Fight Club', 'genres': 'Drama'}
]

# Convert movies to DataFrame
df = pd.DataFrame(movies)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def terminal_recommend():
    print("Available Movies:")
    for movie in movies:
        print(f"{movie['id']}: {movie['title']} ({movie['genres']})")
    movie_id = int(input("Enter a movie ID to get recommendations: "))
    idx = df[df['id'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices][['id', 'title', 'genres']].to_dict('records')
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"{rec['id']}: {rec['title']} ({rec['genres']})")

if __name__ == '__main__':
    terminal_recommend() 