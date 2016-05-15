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

largeDataSet = "../../Large Data/processedWithDates.pickle"
fileAbsolute = "../datasets/training_50000.pickle"

# entireData = pd.read_pickle(fileAbsolute)
entireData = pd.read_pickle("../datasets/groupedByUserAll.pickle")

trainingData = entireData
print trainingData.head(50)
print trainingData.columns

# trainingData.to_pickle("ipython/training_50000.pickle")
trainingData = trainingData[trainingData['duration'] >= 0]
trainingData = trainingData[trainingData['duration'] <= 10000]

# trainingData.loc[:, 'durationShort'] = (trainingData.duration < 5).astype(int)
# trainingData.loc[:, 'durationLong'] = (trainingData.duration >=3600).astype(int)
# trainingData.loc[:, 'durationMid'] = np.invert(np.logical_or(trainingData.durationShort, trainingData.durationLong)).astype(int)

# trainingData.loc[:, 'timeStartBucket'] = trainingData.timeStart.apply(lambda x: x.hour)

# ByPersonByDate = trainingData.groupby(['id', 'date'])

# appleData = trainingData[trainingData['apple'] == 1]
# appleByPerson = appleData.groupby(['id'])
# androidData = trainingData[trainingData['android'] == 1]
# androidByPerson = androidData.groupby('id')

# ByPerson = trainingData.groupby(['id'])

# data generation
# data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))

# print data
# For Apple

def getClusters(dataset):
	identity = dataset.index.get_values()
	# print identity
	# print "\n\n\n"

	durationSeries = dataset['duration']
	wifiSeries = dataset['wifi']
	wirelessSeries = dataset['wireless']
	timeStartSeries = dataset['timeStart']
	weekdaySeries = dataset['weekday']
	numSessionsSeries = dataset['numberSessions']



	data = np.array([list(a) for a in zip(durationSeries, wifiSeries, wirelessSeries, timeStartSeries, weekdaySeries, numSessionsSeries)])
	
	
	# computing K-Means with K = 2 (2 clusters)
	centroids,_ = kmeans(data, 3, iter=200)
	# assign each sample to a cluster
	idx,_ = vq(data,centroids)
	print idx
	print centroids

	# some plotting using numpy's logical indexing
	plot(data[idx==0,0],data[idx==0,5],'ob',
	     data[idx==1,0],data[idx==1,5],'or',
	     data[idx==2,0],data[idx==2,5],'oy')
	plot(centroids[:,0],centroids[:,5],'sg',markersize=8)
	show()

	plot(data[idx==0,1],data[idx==0,4],'ob',
	     data[idx==1,1],data[idx==1,4],'or',
	     data[idx==2,1],data[idx==2,4],'oy')
	plot(centroids[:,1],centroids[:,4],'sg',markersize=8)



	show()


	ax = Axes3D(plt.gcf())
	ax.scatter(data[idx==0,0], data[idx==0,3], data[idx==0,5], zdir='z', s=20, c='b')
	ax.scatter(data[idx==1,0], data[idx==1,3], data[idx==1,5], zdir='z', s=20, c='y')
	ax.scatter(data[idx==2,0], data[idx==2,3], data[idx==2,5], zdir='z', s=20, c='r')
	ax.scatter(centroids[:,0],centroids[:,3], centroids[:, 5], zdir= 'z', s=800, c='g')

	plt.xlabel("Avg Duration")
	ax.set_zlabel("Avg Number Sessions")
	ax.set_ylabel("Avg Session Start Hour")
	plt.title("Top Three Factors from KMeans Clustering Analysis")
	ax.set_xlim3d(0, 11000)
	ax.set_ylim3d(0,24)
	ax.set_zlim3d(0,120)
	plt.savefig("../images/Three factor clustering.png")

	show()
	# For data that have index 1, extract the original data
	# and predict a sequence for them
	# for item in data[idx == 0]:
	# dataset.cluster = data[idx]

	# print identity.shape
	# print data.shape
	# print idx.shape
	# print idx
	data = np.hstack((identity.reshape(len(identity), 1), data, idx.reshape(len(idx), 1)))
	return data


results = getClusters(trainingData)
print results
df = pd.DataFrame({'id':results[:,0],'cluster':results[:,-1]})
# df = df.set_index("id")
print df.head(20)
print trainingData.head(20)
# trainingData = trainingData.merge(df)
merged = trainingData.merge(df.set_index(['id']), left_index=True, right_index=True, how = 'right')

print merged.head(50)

merged.to_pickle("../datasets/mergedWithCluster.pickle")
# Next step is to create markov matrices for each of the clusters
# to confirm or disprove the hypotheses.
# See hypothesis.py

# getClusters(appleByPerson)
# getClusters(androidByPerson)