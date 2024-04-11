import numpy as np
import pandas as pd
movies=pd.read_csv('tmdb_5000_movies.csv/tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv/tmdb_5000_credits.csv')
movies.head(1)
# print(movies.head(1))
credits.head(1)['cast'].values
print(credits.head(1)['cast'].values)

