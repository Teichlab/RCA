# %%
import pandas as pd
import pystan
from sklearn import datasets
import matplotlib.pyplot  as plt
import numpy as np

# %%
stan_model = pystan.StanModel(file='pca.stan')

# %%
iris = datasets.load_iris()
Y = iris.data

# %%
N, G = Y.shape
data = {
    'N': N,
    'G': G,
    'Y': Y
}
o = stan_model.optimizing(data=data)

plt.scatter(o['X'][:, 0], o['X'][:, 1], c='k')
plt.title('{}'.format(o['s2_model']))

# %%
samples = stan_model.sampling(data=data, iter=250, chains=1)

# %%
X = samples.extract('X')['X'].mean(0)
X_var = samples.extract('X')['X'].var(0)

plt.errorbar(X[:, 0], X[:, 1], np.sqrt(X_var[:, 0]), np.sqrt(X_var[:, 1]),
                linestyle='none', marker='o', c='k');
