import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Import linear_kernel
#Import TfIdfVectorizer from scikit-learn
df = pd.read_csv('https://raw.githubusercontent.com/dquarshie89/Data-620/master/movie_plots.csv')

#Rec Based on plot
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['Plot'] = df['Plot'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['Plot'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['Title'].iloc[movie_indices]


get_recommendations('Baby Mama')


#Rec Based on Title
#Construct the required TF-IDF matrix by fitting and transforming the data
title_tfidf_matrix = tfidf.fit_transform(df['Title'])


# Compute the cosine similarity matrix
title_cosine_sim = linear_kernel(title_tfidf_matrix, title_tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

def get_recommendations_title(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    title_sim_scores = list(enumerate(title_cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    title_sim_scores = sorted(title_sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    title_sim_scores = title_sim_scores[1:11]

    # Get the movie indices
    title_movie_indices = [i[0] for i in title_sim_scores]

    # Return the top 10 most similar movies
    return df['Title'].iloc[title_movie_indices]

get_recommendations_title('The Amazing Spider-Man 2')
