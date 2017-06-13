import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('iris_long.csv', index_col=0)

print(data.head())

G = data['variable'].unique().shape[0]
S = data['obs'].unique().shape[0]

N = 2  # Latent space dimensionality

batch_size = 20
num_iter = 3000
learning_rate = 0.01

W = tf.Variable(np.random.randn(G, N), name='weights')
x = tf.Variable(np.random.randn(N, S), name='PCs')

sample_idx = tf.placeholder(tf.int32, shape=[None])
variable_idx = tf.placeholder(tf.int32, shape=[None])
y_ = tf.placeholder(tf.float64, shape=[None])


W_ = tf.gather(W, variable_idx)
x_ = tf.gather(tf.matrix_transpose(x), sample_idx)
y_hat = tf.reduce_sum(W_ * x_, 1)

orth_penalty = tf.abs(tf.reduce_sum(tf.reduce_prod(x_, 1)))

cost = tf.nn.l2_loss(y_ - y_hat) + orth_penalty

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

costs = np.zeros(num_iter)
with tf.Session() as sess:
    sess.run(init)

    for i in range(num_iter):
        batch = data.sample(batch_size)
        feed_dict = {sample_idx: batch['obs'],
                        variable_idx: batch['variable'],
                        y_: batch['value']}

        for j in range(1):
            sess.run(optimizer, feed_dict=feed_dict)

        c = sess.run(cost, feed_dict=feed_dict)
        costs[i] = c
        print('Cost: {}'.format(c))
    
    X_result = sess.run(x)

print(X_result)

import matplotlib.pyplot as plt

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_result[0], X_result[1])
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 2, 2)
# plt.yscale('log')
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')

plt.tight_layout()
plt.show()
