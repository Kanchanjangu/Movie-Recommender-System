import numpy as np
import pandas as pd
import ast
import nltk
import pickle
# Read the movie and credits data
movies = pd.read_csv('tmdb_5000_movies.csv/tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv/tmdb_5000_credits.csv')

# Merge the two DataFrames on 'title'
movies = movies.merge(credits, on='title')

#print(movies.shape)  # Print the shape of the merged DataFrame

# Select necessary columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Check for null values
movies.isnull().sum()#

# Drop rows with null values
movies.dropna(inplace=True)

# Function to convert string representation of list of dictionaries to list of genre names
def convert(obj):
    genres_list = []
    for genre in obj:
        genres_list.append(genre['name'])
    return genres_list

# Apply the conversion function to the 'genres' column
movies['genres'] = movies['genres'].apply(ast.literal_eval).apply(convert)

movies['genres'].iloc[0]#

movies['keywords'] = movies['keywords'].apply(ast.literal_eval).apply(convert)
print(movies.head())
# movies['cast'][0]
movies['cast'] = movies['cast'].apply(ast.literal_eval).apply(convert)

def fetch_director(obj):
    for crew_member in ast.literal_eval(obj):
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan

# Apply the function to fetch director to create a new column 'director'
movies['director'] = movies['crew'].apply(fetch_director)

# Drop the 'crew' column as we have extracted director information
# movies.drop(columns=['crew'], inplace=True)

# Apply lambda function directly to 'overview' column
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Split each genre into separate words within the same list structure
movies['genres'] = movies['genres'].apply(lambda x: [genre.split() for genre in x])
# Apply lambda function to iterate over each list and remove spaces from each string
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") if isinstance(i, str) else i for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") if isinstance(i, str) else i for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") if isinstance(i, str) else i for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") if isinstance(i, str) else i for i in x])

movies['tags']=movies['overview']+ movies['keywords']+movies['genres']+movies['cast']+movies['crew']
new_df=movies[['movie_id','title','tags']]
#print(new_df)
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(map(str, x)))
new_df.head()
new_df['tags'][0]
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
print(new_df.head())
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
print(vectors[0])
# print(cv.get_feature_names_out())
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y) 

new_df['tags'] = new_df['tags'].apply(stem)

# print(new_df['tags'])

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
print(sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6])

# def recommend(movie):
    # movie_index=new_df[new_df['title']==movie].index[0]
    # movies_list=sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6] 
    
    # for i in movies_list:
    #     print(i[0])
# Instead of this:
movies['tags'] = new_df['tags'].apply(lambda x: " ".join(map(str, x)))

# Use this:
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(map(str, x)))
    
    
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances=similarity[movie_index]
    movies_list = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
print(recommend('Batman Begins'))  

pickle.dump(new_df,open('movies.pkl','wb'))  
new_df['title'].values

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))
  