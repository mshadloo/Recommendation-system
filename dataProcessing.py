import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


MOVIELENS_DIR = 'RatingsData'
RATING_DATA_FILE='u.data'
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
#ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), sep='\t', names=r_cols,encoding='latin-1')

# Check the top 5 rows
print(ratings.head())

ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean());
ratings_data=ratings.iloc[:,0:3].values
print(ratings_data[0:5,0:3])
train_data, test_val_data= train_test_split(ratings_data,test_size=0.2)
test_data, val_data=train_test_split(test_val_data,test_size=0.5)


np.savez('ratingsSets100k',train_data=train_data,val_data=val_data,test_data=test_data)

# row_ind = train_data[:,0]
# col_ind = train_data[:,1]
# data = train_data[:,2]
# train_data = sparse.coo_matrix((data, (row_ind, col_ind)))




