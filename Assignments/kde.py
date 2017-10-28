from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

from astropy.table import Table
t = Table().read('joint-bh-mass-table.csv')

x = t['MBH']
x_grid = np.linspace(0,100,100)

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


fig, ax = plt.subplots()
for bandwidth in [0.1, 0.2, 1,1.2,2,2.5,3,5,7]:
    ax.plot(x_grid, kde_sklearn(x, x_grid, bandwidth=bandwidth),
            label='bw={0}'.format(bandwidth), linewidth=3, alpha=0.5)
ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.legend(loc='upper right')


grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 7.0, 100)}, cv=20) # 20-fold cross-validation
grid.fit(x[:, None])
print grid.best_params_

kde = grid.best_estimator_
pdf = np.exp(kde.score_samples(x_grid[:, None]))

fig, ax = plt.subplots()
ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.legend(loc='upper left')
plt.show()
