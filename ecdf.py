from __future__ import division

def ecdf(d):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(d)

    # x-data for the ECDF: x
    x = np.sort(d)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    ind = np.sum(y<.75)

    # return x, y

    # Generate plot
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.axvline(x=x[ind], color='r', linestyle='-')


    # Make the margins nice
    _ = plt.margins(.02)

    # Label the axes
    _ = plt.ylabel('ECDF')
    _ = plt.xlabel('count')

import pandas as pd

data = pd.read_csv('data/train.csv').dropna()

plt.subplot(2, 1, 1)
ecdf(data['Pclass'][data['Survived']==1].values)
plt.subplot(2, 1, 2)
ecdf(data['Pclass'].values)

plt.show()

