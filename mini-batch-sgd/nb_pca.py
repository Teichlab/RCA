import click
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def next_batch(data, batch_size, i, NN):
    indx = (batch_size * i) % NN
    if (batch_size + indx) > NN:
        indx = 1

    return data.iloc[indx:indx + batch_size]


## Model ##
@click.command()
@click.argument('input_file')
@click.option('--obs_col', default='cell')
@click.option('--var_col', default='gene')
@click.option('--val_col', default='expression')
@click.option('--offset_col', default='total_count')
@click.option('--batch_size', default=10000)
@click.option('--num_iter', default=2000)
@click.option('--learning_rate', default=0.01)
@click.option('--inner_iter', default=5)
@click.option('--report_every', default=100)
def main(input_file, obs_col, var_col, val_col, offset_col, batch_size,
         num_iter, learning_rate, inner_iter, report_every):

    ## Data loading ##

    data = pd.read_csv(input_file)

    G = data[var_col].unique().shape[0]
    S = data[obs_col].unique().shape[0]

    tf.logging.info('Shuffling...')
    data = data.sample(frac=1)
    NN = data.shape[0]

    ## CONFIG ##

    N = 2  # Latent space dimensionality

    ## Model ##

    W = tf.Variable(np.random.randn(G, N), name='weights')
    x = tf.Variable(np.random.randn(N, S), name='PCs')
    E = tf.Variable(np.random.randn(S), name='Efficiency')
    S = tf.Variable(np.array([0.]), name='Scaling')

    sample_idx = tf.placeholder(tf.int32, shape=[None])
    variable_idx = tf.placeholder(tf.int32, shape=[None])
    T_ = tf.placeholder(tf.float64, shape=[None])
    y_ = tf.placeholder(tf.float64, shape=[None])


    W_ = tf.gather(W, variable_idx)
    x_ = tf.gather(tf.matrix_transpose(x), sample_idx)
    eta_ = tf.reduce_sum(W_ * x_, 1)
    E_ = tf.gather(E, sample_idx)

    mu_ = tf.exp(eta_ + tf.log(T_) + E_ + S)

    LL = tf.reduce_sum(y_ * tf.log(mu_) - (y_ + 1) * tf.log(mu_ + 1))

    cost = -LL / batch_size

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    costs = np.zeros(num_iter)

    tf.logging.info('Training')
    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_iter):
            batch = next_batch(data, batch_size, i, NN)
            feed_dict = {sample_idx: batch[obs_col],
                        variable_idx: batch[var_col],
                        y_: batch[val_col],
                        T_: batch[offset_col]}

            for j in range(inner_iter):
                sess.run(optimizer, feed_dict=feed_dict)

            c = sess.run(cost, feed_dict=feed_dict)
            costs[i] = c

            if not i % report_every:
                tf.logging.info('Cost: {}'.format(c))
        
        X_result = sess.run(x)
        S_result = sess.run(S)
        E_result = sess.run(E)

    import matplotlib.pyplot as plt

    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X_result[0], X_result[1], s=10, alpha=0.33, c=E_result)
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.subplot(1, 2, 2)
    plt.plot(costs, c='k')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
