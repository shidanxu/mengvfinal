import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn import cross_validation
import input_data
import datetime


def readDataset(fileAbsolute, cutoff = 50000):
	entireData = pd.read_pickle(fileAbsolute)

	trainingData = entireData.head(cutoff)
	print trainingData.head(50)
	# trainingData.to_pickle("ipython/training_50000.pickle")
	trainingData.loc[:, 'apple'] = (trainingData['device'].isin(['I-phone/I-pad', 'iPhone', 'iPad', 'Mac', 'iPod'])).astype(int)
	trainingData.loc[:, 'android'] = (trainingData['device'] == 'android').astype(int)
	trainingData.loc[:, 'duration'] =  (trainingData.timeEnd-trainingData.timeStart).astype('timedelta64[s]')
	trainingData.loc[:, 'durationShort'] = (trainingData.duration < 5).astype(int)
	trainingData.loc[:, 'durationLong'] = (trainingData.duration >=3600).astype(int)
	trainingData.loc[:, 'durationMid'] = np.invert(np.logical_or(trainingData.durationShort, trainingData.durationLong)).astype(int)

	trainingData.loc[:, 'timeStartBucket'] = trainingData.timeStart.apply(lambda x: x.hour)
	
	ByPersonByDate = trainingData.groupby(['id', 'date'])

	numSessions = ByPersonByDate.count()['timeStart']

	print type(numSessions)
	print numSessions

	return trainingData

def datasetToTrainTest(data, x_columns, y_columns, test_percentage = 0.4):
	x_data = data.as_matrix(columns = x_columns).astype(np.float32)
	y_data = data.as_matrix(columns = y_columns).astype(np.float32)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=test_percentage, random_state=0)

	return X_train, X_test, y_train, y_test

if __name__ == '__main__':
	trainingData = readDataset("../datasets/training_50000.pickle")
	xtr, xte, ytr, yte = datasetToTrainTest(trainingData, x_columns = ['apple', 'android', 'timeStartBucket', 'weekday'],
		y_columns = ['durationShort', 'durationMid', 'durationLong'])

	
