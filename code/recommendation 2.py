import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.sparse.linalg import svds
from categories import category_dicts
##좋아요 기반 추천
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

movie_data = pd.read_csv('movies_metadata.csv',  dtype={'movieId': str, 'original_language': str, 'vote_count': float}, low_memory=False)
movie_data =  movie_data.loc[movie_data['original_language'] == 'en', :]
movie_data = movie_data[['movieId', 'title', 'original_language', 'genres']]

movie_keyword = pd.read_csv('keywords.csv',encoding='utf-8')

movie_data['movieId'] = movie_data['movieId'].astype(int)
movie_keyword['movieId'] = movie_keyword['movieId'].astype(int)
movie_data = pd.merge(movie_data, movie_keyword, on='movieId')

movie_data['genres'] = movie_data['genres'].apply(literal_eval)
movie_data['genres'] = movie_data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))
movie_data['keywords'] = movie_data['keywords'].apply(literal_eval)
movie_data['keywords'] = movie_data['keywords'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

print(movie_data.head())

tfidf_vector = TfidfVectorizer()
#tfidf_vector = TfidfVectorizer(ngram_range=(1,2))
tfidf_matrix = tfidf_vector.fit_transform(movie_data['genres'] + " " + movie_data['keywords']).toarray()
#tfidf_matrix = tfidf_vector.fit_transform(movie_data['genres']).toarray()
tfidf_matrix_feature = tfidf_vector.get_feature_names_out()


tfidf_matrix = pd.DataFrame(tfidf_matrix, columns=tfidf_matrix_feature, index = movie_data.title)
#print(tfidf_matrix.shape)
tfidf_matrix.head()

#m = movie_data['vote_count'].quantile(0.9)
#movie_data = movie_data.loc[movie_data['vote_count'] >= m]
#C = movie_data['vote_average'].mean()

#print(C)
#print(m)


#def weighted_rating(x, m=m, C=C):
#    v = x['vote_count']
#    R = x['vote_average']
#    return ( v / (v+m) * R ) + (m / (m + v) * C)





def combined_recommendation(user_id, tfidf_matrix, svd_predictions, movie_titles, num_recommendations=10, alpha=0.5, beta=0.5):
    user_index = user_id - 1
    
    # Get SVD-based predictions
    svd_scores = svd_predictions[user_index]
    
    # Get TF-IDF similarity scores
    tfidf_scores = tfidf_matrix[user_index]

    # Combine scores with weights
    combined_scores = alpha * tfidf_scores + beta * svd_scores
    
    # Create a DataFrame for recommendations
    recommendations = pd.DataFrame({
        'movie_title': movie_titles,
        'combined_score': combined_scores
    }).sort_values(by='combined_score', ascending=False)
    
    return recommendations.head(num_recommendations)
