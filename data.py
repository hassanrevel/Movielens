import pandas as pd
import tensorflow_datasets as tfds
from sklearn.utils import shuffle
import pickle
from tqdm.auto import tqdm

# Fetching the data

# Data
ds = tfds.load('movielens/latest-small-ratings')

# Initialize lists to store data
user_ids = []
movie_ids = []
ratings = []

# Iterate over the dataset and extract desired data
for example in tqdm(ds['train'], total=len(ds['train']), desc='Parsing'):
    user_id = int(example['user_id'].numpy().decode())      # ID of the user
    movie_id = int(example['movie_id'].numpy().decode())    # ID of the movie
    rating = example['user_rating'].numpy()                 # Rating given by the user

    # Append data to lists
    user_ids.append(user_id)
    movie_ids.append(movie_id)
    ratings.append(rating)

# Create a DataFrame
df = pd.DataFrame({'userId': user_ids, 'movieId': movie_ids, 'rating': ratings})

# Processing the data

df.userId = df.userId - 1

unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count
  count += 1

df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

## The N
N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

## Data Split
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

## Train
print('Preparing the train ....')

user2movie = {} # Set of all movies whom user i rated, sigma
movie2user = {} # Set of all users who rated movie j, si
usermovie2rating = {}

def update_user2movie_and_movie2user(row):
  i = int(row.userId)
  j = int(row.movie_idx)
  if i not in user2movie:
    user2movie[i] = [j]
  else:
    user2movie[i].append(j)

  if j not in movie2user:
    movie2user[j] = [i]
  else:
    movie2user[j].append(i)

  usermovie2rating[(i,j)] = row.rating
df_train.apply(update_user2movie_and_movie2user, axis=1)

print('Length of user2movie:', len(user2movie))
print('Length of movie2user:', len(movie2user))
print('Length of usermovie2rating:', len(usermovie2rating))

## Test
print('Preparing the test ...')

usermovie2rating_test = {}
def update_usermovie2rating_test(row):
  i = int(row.userId)
  j = int(row.movieId)
  usermovie2rating_test[(i,j)] = row.rating
df_test.apply(update_usermovie2rating_test, axis=1)

print('Length of usermovie2rating_test:', len(usermovie2rating_test))

# Saving
df.to_csv('data/movielens_subset/movielens_subset.csv', index=False)

with open('data/movielens_subset/user2movie.json', 'wb') as f:
  pickle.dump(user2movie, f)

with open('data/movielens_subset/movie2user.json', 'wb') as f:
  pickle.dump(movie2user, f)

with open('data/movielens_subset/usermovie2rating.json', 'wb') as f:
  pickle.dump(usermovie2rating, f)

with open('data/movielens_subset/usermovie2rating_test.json', 'wb') as f:
  pickle.dump(usermovie2rating_test, f)