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

# Calculating weights
K = 20
limit = 5
averages = []
deviations = []
neighbors = []


for j in tqdm(range(M), total=M, desc='Training Weights'):

  ########### Item j ###############

  users_j = movie2user.get(j) # contains users who rated j, sigma_j
  if users_j != None:
    users_j_set = set(users_j) # set of users who rated j, set(sigma_j)

    ratings_ij = {user:usermovie2rating[(user, j)] for user in users_j} # ratings that set of users gave to j, rij

    ratings_avg_j = np.mean(list(ratings_ij.values())) # rj-

    ratings_dev_j = {user:(rating - ratings_avg_j) for user, rating in ratings_ij.items()} # {rij - rj-}
    ratings_dev_j_values = np.array(list(ratings_dev_j.values()))

    sigma_j = np.sqrt(ratings_dev_j_values.dot(ratings_dev_j_values))

    averages.append(ratings_avg_j)
    deviations.append(ratings_dev_j)

    sl = SortedList()
    for jbar in range(M):
      if jbar != j:

        ########### Rest of the items ##################

        users_jbar = movie2user.get(jbar)
        if users_jbar != None:
          users_jbar_set = set(users_jbar) # set of users who rated item j`

          common_users = (users_j_set & users_jbar_set)

          if len(common_users) > limit:
            ratings_jbar = {user:usermovie2rating[(user, jbar)] for user in users_jbar} # rij`
            ratings_avg_jbar = np.mean(list(ratings_jbar.values()))

            ratings_dev_jbar = {user:(rating_jbar - ratings_avg_jbar) for user, rating_jbar in ratings_jbar.items()} # rij` - rj`
            ratings_dev_jbar_values = np.array(list(ratings_dev_jbar.values()))

            sigma_jbar = np.sqrt(ratings_dev_jbar_values.dot(ratings_dev_jbar_values))

            numerator = sum(ratings_dev_j[m] * ratings_dev_jbar[m] for m in common_users)

            denominator = (sigma_j * sigma_jbar)

            w_jjbar = numerator / denominator

            sl.add((-w_jjbar, jbar))

  sl = sl[:K]
  neighbors.append(sl)

# Predict Function
def predict(i, j):

  numerator = 0
  denominator = 0

  for neg_w, jbar in neighbors[j]: # Neighbors weights and items
    try:
      numerator += -neg_w * deviations[jbar][i] # Deviation of the particular item j` rated by user i with rest of items
      denominator += np.abs(neg_w)
    except (IndexError, KeyError):
      pass

  if denominator == 0:
    try:
      prediction = averages[j]
    except IndexError:
      return 0
  else:
    try:
      prediction = averages[j] + numerator/denominator
    except IndexError:
      return 0

  # We want the rating to be between 0.5 and 5
  prediction = min(5, prediction)
  prediction = max(0.5, prediction)

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