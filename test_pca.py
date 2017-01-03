# %%
import pandas as pd
import pystan
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot  as plt
import numpy as np

plt.style.use('seaborn-notebook')
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.figsize'] = [5, 4]
plt.rcParams['axes.linewidth'] = 1.

# %%
pca = pystan.StanModel(file='pca.stan')

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
opt = pca.optimizing(data=data)
o = opt
min_noise = opt['s2_model']

df = pd.DataFrame(Y, columns=iris.feature_names)
df['X1'] = o['X'][:, 0]
df['X2'] = o['X'][:, 1]
df['species'] = iris.target_names[iris.target]

colors = {'setosa':     (33./256, 33./256, 33./256),
          'versicolor': (48./256, 108./256, 130./256),
          'virginica':  (181./256, 64./256, 39./256)}

for name, group in df.groupby('species'):
    plt.scatter(group['X1'], group['X2'],
    s=30,
    label=name,
    edgecolor='none',
    color=colors[name])

print(min_noise)

plt.xlabel('$X_1$ (PC1)')
plt.ylabel('$X_2$ (PC2)')

plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

plt.legend(loc='upper center',
           scatterpoints=3,
           bbox_to_anchor=(0.5, -0.17),
           ncol=3)

plt.title('Principal Component Analysis\n(PML Estimate)')
plt.savefig('iris_pca.png', bbox_inches='tight')

# %%
pd.DataFrame(Y).describe()

# %%
batch = np.random.binomial(1, 0.5, (Y.shape[0], 1))
effect = np.random.normal(2.0, 0.5, size=Y.shape)
Y_b = Y + batch * effect

# %%
N, G = Y_b.shape
data = {
    'N': N,
    'G': G,
    'Y': Y_b
}

opt = pca.optimizing(data=data)
o = opt
min_noise = opt['s2_model']

df = pd.DataFrame(Y_b, columns=iris.feature_names)
df['X1'] = o['X'][:, 0]
df['X2'] = o['X'][:, 1]
df['species'] = iris.target_names[iris.target]
df['batch'] = batch

marker = {0: '^', 1: '*'}

for name, group in df.groupby(['batch', 'species']):
    plt.scatter(group['X1'], group['X2'],
    s=40 + 30 * name[0],
    label='Batch {} - {}'.format(name[0] + 1, name[1]),
    edgecolor='none',
    marker=marker[name[0]],
    color=colors[name[1]])

print(min_noise)

plt.xlabel('$X_1$ (PC1)')
plt.ylabel('$X_2$ (PC2)')

plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

plt.legend(loc='upper center',
           scatterpoints=3,
           bbox_to_anchor=(0.5, -0.17),
           ncol=2)

plt.title('Principal Component Analysis\n(PML Estimate)')
plt.savefig('iris_batch_pca.png', bbox_inches='tight')

# %%
rca = pystan.StanModel(file='rca.stan')

# %%
N, G = Y_b.shape
data = {
    'N': N,
    'G': G,
    'Y': Y_b,
    'P': 1,
    'Z': batch
}

opt = rca.optimizing(data=data)
o = opt
min_noise = opt['s2_model']

df = pd.DataFrame(Y_b, columns=iris.feature_names)
df['X1'] = o['X'][:, 0]
df['X2'] = o['X'][:, 1]
df['species'] = iris.target_names[iris.target]
df['batch'] = batch

for name, group in df.groupby(['batch', 'species']):
    plt.scatter(group['X1'], group['X2'],
    s=40 + 30 * name[0],
    label='Batch {} - {}'.format(name[0] + 1, name[1]),
    edgecolor='none',
    marker=marker[name[0]],
    color=colors[name[1]])

plt.xlabel('$X_1$ (PC1)')
plt.ylabel('$X_2$ (PC2)')

plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

plt.legend(loc='upper center',
           scatterpoints=3,
           bbox_to_anchor=(0.5, -0.17),
           ncol=2)

plt.title('Residual Component Analysis\n(PML Estimate)')

print(o['s2_model'])
plt.savefig('iris_batch_rca.png', bbox_inches='tight')


# %% 
plt.rcParams
