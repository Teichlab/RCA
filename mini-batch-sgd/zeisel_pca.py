import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('zeisel_sample_long.csv', index_col=0)

print(data.head())

G = data['gene'].unique().shape[0]
S = data['cell'].unique().shape[0]

print('Shuffling...')
data = data.sample(frac=1)
NN = data.shape[0]

def next_batch(data, batch_size, i):
    indx = (batch_size * i) % NN
    if (batch_size + indx) > NN:
        indx = 1

    return data.iloc[indx:indx + batch_size]

N = 2  # Latent space dimensionality

batch_size = 10000
num_iter = 1000
learning_rate = 0.01

W = tf.Variable(np.random.randn(G, N), name='weights')
x = tf.Variable(np.random.randn(N, S), name='PCs')

sample_idx = tf.placeholder(tf.int32, shape=[None])
variable_idx = tf.placeholder(tf.int32, shape=[None])
y_ = tf.placeholder(tf.float64, shape=[None])


W_ = tf.gather(W, variable_idx)
x_ = tf.gather(tf.matrix_transpose(x), sample_idx)
y_hat = tf.reduce_sum(W_ * x_, 1)

cost = tf.nn.l2_loss(y_ - y_hat) / batch_size

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

costs = np.zeros(num_iter)

config = tf.ConfigProto(device_count={'CPU': 4},
                        inter_op_parallelism_threads=4,
                        intra_op_parallelism_threads=1)
config.graph_options.optimizer_options.opt_level = -1

print('Training')
with tf.Session(config=config) as sess:
    sess.run(init)

    for i in range(num_iter):
        batch = next_batch(data, batch_size, i)
        feed_dict = {sample_idx: batch['cell'],
                     variable_idx: batch['gene'],
                     y_: batch['expression']}

        for j in range(3):
            sess.run(optimizer, feed_dict=feed_dict)

        c = sess.run(cost, feed_dict=feed_dict)
        costs[i] = c

        if not i % 100:
            print('Cost: {}'.format(c))
    
    X_result = sess.run(x)

print(X_result)

import matplotlib.pyplot as plt

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_result[0], X_result[1], s=10, alpha=0.33, c='k')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 2, 2)
# plt.yscale('log')
plt.plot(costs, c='k')
plt.xlabel('Iteration')
plt.ylabel('Cost')

plt.tight_layout()
plt.show()
