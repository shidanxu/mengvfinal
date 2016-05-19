from pylab import plot,show
from numpy import vstack,array,hstack
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
from sklearn import cross_validation
import datetime


df = pd.read_pickle("../datasets/mergedWithCluster.pickle")

grouped = df.groupby("cluster").mean()

counted = df.groupby("cluster").count()
print "Counted: \n"
print counted

grouped.loc['AVG'] = [13.573331, 14.086646, 2.951581, 41.712824, -74.101604, 0.274385, 0.725615, 1847.931331, 20.645197]
print grouped
grouped['numberSessions'].plot(kind = 'bar')
plt.xlabel("Cluster")
plt.ylabel("Avg. Daily Number of Sessions")

plt.show()

print df.mean()