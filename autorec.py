import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from scipy.sparse import lil_matrix
import tensorflow.keras.backend as K

# Load in the data
df = pd.read_csv('data/movielens_subset/movielens_subset.csv')

N = df.userId.max() + 1  # number of users
M = df.movie_idx.max() + 1  # number of movies

# Data Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Converting Train Matrix into sparse
print('Converting train matrix to sparse ...')
A_train = lil_matrix((N, M))

def update_train(row):
    i = int(row.userId)
    j = int(row.movie_idx)
    A_train[i, j] = row.rating

df_train.apply(update_train, axis=1)
A_train = A_train.tocsr()

# Convert Test Matrix into sparse
print('Converting test matrix to sparse ...')
A_test = lil_matrix((N, M))

def update_test(row):
    i = int(row.userId)
    j = int(row.movie_idx)
    A_test[i, j] = row.rating

df_test.apply(update_test, axis=1)
A_test = A_test.tocsr()

mask_train = (A_train > 0).astype(float)
mask_test = (A_test > 0).astype(float)

## make copies since we will shuffle
A_train_copy = A_train.copy()
mask_train_copy = mask_train.copy()
A_test_copy = A_test.copy()
mask_test_copy = mask_test.copy()

# Calculate the global mean rating
mu = A_train.sum() / mask_train.sum()

# Initializations
run = 5
epochs = 10
batch_size = 32
reg = 0.0001

# Model
i = Input(shape=(M,))
x = Dropout(0.7)(i)
x = Dense(700, activation='tanh', kernel_regularizer=l2(reg))(x)
x = Dense(M, kernel_regularizer=l2(reg))(x)

def custom_loss(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
    diff = y_pred - y_true
    sqdiff = diff * diff * mask
    sse = K.sum(K.sum(sqdiff))
    n = K.sum(K.sum(mask))
    return sse / n

def train_generator(A, M):
    while True:
        A, M = shuffle(A, M)
        for i in range(A.shape[0] // batch_size + 1):
            upper = min((i + 1) * batch_size, A.shape[0])
            a = A[i * batch_size:upper].toarray()
            m = M[i * batch_size:upper].toarray()
            a = a - mu * m  # must keep zeros at zero!
            noisy = a  # no noise
            yield noisy, a

def test_generator(A, M, A_test, M_test):
    while True:
        for i in range(A.shape[0] // batch_size + 1):
            upper = min((i + 1) * batch_size, A.shape[0])
            a = A[i * batch_size:upper].toarray()
            m = M[i * batch_size:upper].toarray()
            at = A_test[i * batch_size:upper].toarray()
            mt = M_test[i * batch_size:upper].toarray()
            a = a - mu * m
            at = at - mu * mt
            yield a, at

model = Model(i, x)
model.compile(
    loss=custom_loss,
    optimizer=SGD(learning_rate=0.08, momentum=0.9),
    metrics=[custom_loss]
)
history = model.fit(
    train_generator(A_train, mask_train),
    validation_data=test_generator(A_train_copy, mask_train_copy, A_test_copy, mask_test_copy),
    epochs=epochs,
    steps_per_epoch=A_train.shape[0] // batch_size + 1,
    validation_steps=A_test.shape[0] // batch_size + 1
)

# Runs
runs_dir = f'runs/run{run}'
os.makedirs(runs_dir, exist_ok=True)

# Plot losses
r = history.history

loss_plot_save_path = f'{runs_dir}/loss.png'
os.makedirs(os.path.dirname(loss_plot_save_path), exist_ok=True)
plt.plot(r['loss'], label='train loss')
plt.plot(r['val_loss'], label='val loss')
plt.legend()
plt.savefig(loss_plot_save_path)
print('Loss Plot had been saved at', loss_plot_save_path)

mse_plot_save_path = f'{runs_dir}/mse.png'
os.makedirs(os.path.dirname(loss_plot_save_path), exist_ok=True)
plt.plot(r['custom_loss'], label='train mse')
plt.plot(r['val_custom_loss'], label='val mse')
plt.legend()
plt.savefig(mse_plot_save_path)
print('MSE Plot had been saved at', mse_plot_save_path)

# Saving the Metrics
metrics_save_path = f'{runs_dir}/metrics.csv'
os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
pd.DataFrame(r).to_csv(metrics_save_path, index=False)
print('Metrics had been saved at', metrics_save_path)

# Saving the model
model_save_path = f'{runs_dir}/model.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print('Model had been saved at', model_save_path)
