import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten, \
    Dropout, BatchNormalization, Activation, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD

# Load in the data
df = pd.read_csv('data/movielens_subset/movielens_subset.csv')

N = df.userId.max() + 1
M = df.movie_idx.max() + 1

# Data Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Initializations
run = 4
K = 10
mu = df_train.rating.mean()
epochs = 10
batch_size = 32
reg = 0.

# Model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u)
m_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(m)

## Main branch
### Note: Loss is lower if you don't use this main branch and don't do merging
u_bias = Embedding(N, 1)(u) # (N, 1, 1)
m_bias = Embedding(M, 1)(m) # (N, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding])
x = Add()([x, u_bias, m_bias])
x = Flatten()(x)

## Side Branch
u_embedding = Flatten()(u_embedding)
m_embedding = Flatten()(m_embedding)
y = Concatenate()([u_embedding, m_embedding])
y = Dense(400)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(1)(y)

## Merge
x = Add()([x, y])

model = Model(inputs=[u, m], outputs=x)
model.compile(
    loss='mse',
    optimizer=SGD(lr=0.01, momentum=0.9),
    metrics=['mse']
)
history = model.fit(
    x = [df_train.userId.values, df_train.movie_idx.values],
    y = df_train.rating.values - mu,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (
        [df_test.userId.values, df_test.movie_idx.values],
        df_test.rating.values - mu
    )
)

# Runs
runs_dir = f'runs/run{run}'
os.makedirs(runs_dir, exist_ok=True)

# plot losses
r = history.history

loss_plot_save_path = f'{runs_dir}/loss.png'
os.makedirs(os.path.dirname(loss_plot_save_path), exist_ok=True)
plt.plot(r['loss'], label='train loss')
plt.plot(r['val_loss'], label='val loss')
plt.legend()
plt.savefig(loss_plot_save_path);
print('Loss Plot had been saved at', loss_plot_save_path)


# Plot MSE
mse_plot_save_path = f'{runs_dir}/mse.png'
os.makedirs(os.path.dirname(mse_plot_save_path), exist_ok=True)
plt.plot(r['mse'], label='train mse')
plt.plot(r['val_mse'], label='val mse')
plt.legend()
plt.savefig(mse_plot_save_path);

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


