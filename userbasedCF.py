import pickle
import numpy as np
from tqdm.auto import tqdm
from sortedcontainers import SortedList
from sklearn.metrics import mean_squared_error

with open('data/movielens_subset/user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open('data/movielens_subset/movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open('data/movielens_subset/usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('data/movielens_subset/usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)


N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1

K = 25 # number of neighbors we'd like to consider
limit = 5 # number of common movies users must have in common in order to consider
neighbors = [] # store neighbors in this list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use

for i in tqdm(range(N), total=N, desc='Computing Weights'):
  # find the 25 closest users to user i
  movies_i = user2movie[i]
  movies_i_set = set(movies_i)

  # calculate avg and deviation
  ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { movie:(rating - avg_i) for movie, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # save these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(N):
    # don't include yourself
    if j != i:
      movies_j = user2movie[j]
      movies_j_set = set(movies_j)
      common_movies = (movies_i_set & movies_j_set) # intersection
      if len(common_movies) > limit:
        # calculate avg and deviation
        ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # calculate correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
        w_ij = numerator / (sigma_i * sigma_j)

        # insert into sorted list and truncate
        # negate weight, because list is sorted ascending
        # maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:
          del sl[-1]

  # store the neighbors
  neighbors.append(sl)

def predict(i, m):
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # remember, the weight is stored as its negative
    # so the negative of the negative weight is the positive weight
    try:
      numerator += -neg_w * deviations[j][m]
      denominator += abs(neg_w)
    except KeyError:
      # neighbor may not have rated the same movie
      # don't want to do dictionary lookup twice
      # so just throw exception
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # min rating is 0.5
  return prediction


train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
  prediction = predict(i, m)
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
for (i, m), target in usermovie2rating_test.items():
  prediction = predict(i, m)
  test_predictions.append(prediction)
  test_targets.append(target)


print('train mse:', mean_squared_error(train_predictions, train_targets))
print('test mse:', mean_squared_error(test_predictions, test_targets))

