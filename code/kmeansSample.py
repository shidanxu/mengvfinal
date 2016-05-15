from pylab import plot,show
from numpy import vstack,array,hstack
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import pickle
from sklearn import cross_validation
import datetime

largeDataSet = "../../Large Data/processedWithDates.pickle"
fileAbsolute = "../datasets/training_50000.pickle"

# entireData = pd.read_pickle(fileAbsolute)
entireData = pd.read_pickle(largeDataSet)

trainingData = entireData.head(500000)
print trainingData.head(50)

# trainingData.to_pickle("ipython/training_50000.pickle")
trainingData.loc[:, 'apple'] = (trainingData['device'].isin(['I-phone/I-pad', 'iPhone', 'iPad', 'Mac', 'iPod'])).astype(int)
trainingData.loc[:, 'android'] = (trainingData['device'] == 'android').astype(int)
trainingData.loc[:, 'duration'] =  (trainingData.timeEnd-trainingData.timeStart).astype('timedelta64[s]')
trainingData = trainingData[trainingData['duration'] >= 0]
trainingData = trainingData[trainingData['duration'] <= 3000]

# trainingData.loc[:, 'durationShort'] = (trainingData.duration < 5).astype(int)
# trainingData.loc[:, 'durationLong'] = (trainingData.duration >=3600).astype(int)
# trainingData.loc[:, 'durationMid'] = np.invert(np.logical_or(trainingData.durationShort, trainingData.durationLong)).astype(int)

trainingData.loc[:, 'timeStartBucket'] = trainingData.timeStart.apply(lambda x: x.hour)

# ByPersonByDate = trainingData.groupby(['id', 'date'])

appleData = trainingData[trainingData['apple'] == 1]
appleByPerson = appleData.groupby(['id'])
androidData = trainingData[trainingData['android'] == 1]
androidByPerson = androidData.groupby('id')

ByPerson = trainingData.groupby(['id'])

# data generation
# data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))

# print data
# For Apple

def getClusters(dataset):
	durationSeries = dataset.mean()['duration']
	appleSeries = dataset.mean()['apple']
	androidSeries = dataset.mean()['android']
	timeStartSeries = dataset.mean()['timeStartBucket']

	data = np.array([list(a) for a in zip(durationSeries, appleSeries, androidSeries, timeStartSeries)])
	print data
	# print np.array(data)

	# computing K-Means with K = 2 (2 clusters)
	centroids,_ = kmeans(data)
	# assign each sample to a cluster
	idx,_ = vq(data,centroids)
	print idx
	print centroids

	# some plotting using numpy's logical indexing
	plot(data[idx==0,0],data[idx==0,3],'ob',
	     data[idx==1,0],data[idx==1,3],'or',
	     data[idx==2,0],data[idx==2,3],'oy')
	plot(centroids[:,0],centroids[:,3],'sg',markersize=8)
	show()

	plot(data[idx==0,1],data[idx==0,2],'ob',
	     data[idx==1,1],data[idx==1,2],'or',
	     data[idx==2,1],data[idx==2,2],'oy')
	plot(centroids[:,1],centroids[:,2],'sg',markersize=8)


	show()

	# For data that have index 1, extract the original data
	# and predict a sequence for them
	for item in data[idx == 0]:



getClusters(appleByPerson)
getClusters(androidByPerson)