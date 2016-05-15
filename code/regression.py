import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
from sklearn import cross_validation

import matplotlib.pyplot as plt





# entireData = pd.read_pickle("ipython/training_50000.pickle")

# trainingData = entireData.head(50000)
# # trainingData.to_pickle("ipython/training_50000.pickle")
# trainingData['apple'] = (trainingData['device'].isin(['I-phone/I-pad', 'iPhone', 'iPad', 'Mac', 'iPod'])).astype(int)
# trainingData['android'] = (trainingData['device'] == 'android').astype(int)
# trainingData['duration'] =  (trainingData.timeEnd-trainingData.timeStart).astype('timedelta64[s]')
# trainingData['durationShort'] = (trainingData.duration < 5).astype(int)
# trainingData['durationLong'] = (trainingData.duration >=3600).astype(int)
# trainingData['durationMid'] = np.invert(np.logical_or(trainingData.durationShort, trainingData.durationLong)).astype(int)

# trainingData['timeStartBucket'] = trainingData.timeStart.apply(lambda x: x.hour)
# trainingData['ip1'] = trainingData.ip.apply(lambda x: bin(int(x.split(".")[0]))[2:].zfill(8))
# trainingData['ip2'] = trainingData.ip.apply(lambda x: bin(int(x.split(".")[1]))[2:].zfill(8))
# trainingData['ip3'] = trainingData.ip.apply(lambda x: bin(int(x.split(".")[2]))[2:].zfill(8))
# trainingData['ip4'] = trainingData.ip.apply(lambda x: bin(int(x.split(".")[3]))[2:].zfill(8))

# trainingData['ip01'] = trainingData.ip1.apply(lambda x: x[0]).astype(int)
# trainingData['ip02'] = trainingData.ip1.apply(lambda x: x[1]).astype(int)
# trainingData['ip03'] = trainingData.ip1.apply(lambda x: x[2]).astype(int)
# trainingData['ip04'] = trainingData.ip1.apply(lambda x: x[3]).astype(int)
# trainingData['ip05'] = trainingData.ip1.apply(lambda x: x[4]).astype(int)
# trainingData['ip06'] = trainingData.ip1.apply(lambda x: x[5]).astype(int)
# trainingData['ip07'] = trainingData.ip1.apply(lambda x: x[6]).astype(int)
# trainingData['ip08'] = trainingData.ip1.apply(lambda x: x[7]).astype(int)

# trainingData['ip09'] = trainingData.ip2.apply(lambda x: x[0]).astype(int)
# trainingData['ip10'] = trainingData.ip2.apply(lambda x: x[1]).astype(int)
# trainingData['ip11'] = trainingData.ip2.apply(lambda x: x[2]).astype(int)
# trainingData['ip12'] = trainingData.ip2.apply(lambda x: x[3]).astype(int)
# trainingData['ip13'] = trainingData.ip2.apply(lambda x: x[4]).astype(int)
# trainingData['ip14'] = trainingData.ip2.apply(lambda x: x[5]).astype(int)
# trainingData['ip15'] = trainingData.ip2.apply(lambda x: x[6]).astype(int)
# trainingData['ip16'] = trainingData.ip2.apply(lambda x: x[7]).astype(int)

# trainingData['ip17'] = trainingData.ip3.apply(lambda x: x[0]).astype(int)
# trainingData['ip18'] = trainingData.ip3.apply(lambda x: x[1]).astype(int)
# trainingData['ip19'] = trainingData.ip3.apply(lambda x: x[2]).astype(int)
# trainingData['ip20'] = trainingData.ip3.apply(lambda x: x[3]).astype(int)
# trainingData['ip21'] = trainingData.ip3.apply(lambda x: x[4]).astype(int)
# trainingData['ip22'] = trainingData.ip3.apply(lambda x: x[5]).astype(int)
# trainingData['ip23'] = trainingData.ip3.apply(lambda x: x[6]).astype(int)
# trainingData['ip24'] = trainingData.ip3.apply(lambda x: x[7]).astype(int)

# trainingData['ip25'] = trainingData.ip4.apply(lambda x: x[0]).astype(int)
# trainingData['ip26'] = trainingData.ip4.apply(lambda x: x[1]).astype(int)
# trainingData['ip27'] = trainingData.ip4.apply(lambda x: x[2]).astype(int)
# trainingData['ip28'] = trainingData.ip4.apply(lambda x: x[3]).astype(int)
# trainingData['ip29'] = trainingData.ip4.apply(lambda x: x[4]).astype(int)
# trainingData['ip30'] = trainingData.ip4.apply(lambda x: x[5]).astype(int)
# trainingData['ip31'] = trainingData.ip4.apply(lambda x: x[6]).astype(int)
# trainingData['ip32'] = trainingData.ip4.apply(lambda x: x[7]).astype(int)


# x_data = trainingData.as_matrix(columns = ['apple', 'android', 'timeStartBucket', 'weekday', 'ip01','ip02','ip03','ip04','ip05','ip06','ip07','ip08',
# 	'ip09','ip10','ip11','ip12','ip13','ip14','ip15','ip16','ip17','ip18','ip19','ip20','ip21','ip22','ip23','ip24',
# 	'ip25','ip26','ip27','ip28','ip29','ip30','ip31','ip32']).astype(np.float32)
# # Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
# y_data = trainingData.as_matrix(columns = ['duration']).astype(np.float32)

# print "X, Y_data initialized!!"
# clf = linear_model.Ridge(alpha = .5)

# logitReg = linear_model.LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.4, random_state=0)

# clf.fit(X_train, y_train)
# logitReg.fit(X_train, y_train)

# print clf.coef_
# print logitReg.coef_

# plt.plot(clf.coef_[0], "+")
# plt.title("Ridge Regression")

# plt.savefig("images/ridgeRegression.png")
# plt.show()

# print "Training set score is: ", clf.score(X_train, y_train)
# print "Testing set score is: ", clf.score(X_test, y_test)

if __name__ == '__main__':
	entireData = pd.read_pickle("ipython/processedWithDates.pickle")

	trainingData = entireData.head(1000000)
	# trainingData.to_pickle("ipython/training_50000.pickle")
	trainingData.loc[:,'apple'] = (trainingData['device'].isin(['I-phone/I-pad', 'iPhone', 'iPad', 'Mac', 'iPod'])).astype(int)
	trainingData.loc[:,'android'] = (trainingData['device'] == 'android').astype(int)
	trainingData.loc[:,'duration'] =  (trainingData.timeEnd-trainingData.timeStart).astype('timedelta64[s]')
	trainingData.loc[:,'durationShort'] = (trainingData.duration < 5).astype(int)
	trainingData.loc[:,'durationLong'] = (trainingData.duration >=3600).astype(int)
	trainingData.loc[:,'durationMid'] = np.invert(np.logical_or(trainingData.durationShort, trainingData.durationLong)).astype(int)

	def durationY(duration):
		if duration < 5:
			return 'short'
		elif duration < 3600:
			return 'mid'
		else:
			return 'long'
	
	trainingData.loc[:,'durationY'] = trainingData.duration.apply(durationY)

	trainingData.loc[:,'timeStartBucket'] = trainingData.timeStart.apply(lambda x: x.hour)
	trainingData.loc[:,'ip1'] = trainingData.ip.apply(lambda x: bin(int(x.split(".")[0]))[2:].zfill(8))
	trainingData.loc[:,'ip2'] = trainingData.ip.apply(lambda x: bin(int(x.split(".")[1]))[2:].zfill(8))
	trainingData.loc[:,'ip3'] = trainingData.ip.apply(lambda x: bin(int(x.split(".")[2]))[2:].zfill(8))
	trainingData.loc[:,'ip4'] = trainingData.ip.apply(lambda x: bin(int(x.split(".")[3]))[2:].zfill(8))

	trainingData.loc[:,'ip01'] = trainingData.ip1.apply(lambda x: x[0]).astype(int)
	trainingData.loc[:,'ip02'] = trainingData.ip1.apply(lambda x: x[1]).astype(int)
	trainingData.loc[:,'ip03'] = trainingData.ip1.apply(lambda x: x[2]).astype(int)
	trainingData.loc[:,'ip04'] = trainingData.ip1.apply(lambda x: x[3]).astype(int)
	trainingData.loc[:,'ip05'] = trainingData.ip1.apply(lambda x: x[4]).astype(int)
	trainingData.loc[:,'ip06'] = trainingData.ip1.apply(lambda x: x[5]).astype(int)
	trainingData.loc[:,'ip07'] = trainingData.ip1.apply(lambda x: x[6]).astype(int)
	trainingData.loc[:,'ip08'] = trainingData.ip1.apply(lambda x: x[7]).astype(int)

	trainingData.loc[:,'ip09'] = trainingData.ip2.apply(lambda x: x[0]).astype(int)
	trainingData.loc[:,'ip10'] = trainingData.ip2.apply(lambda x: x[1]).astype(int)
	trainingData.loc[:,'ip11'] = trainingData.ip2.apply(lambda x: x[2]).astype(int)
	trainingData.loc[:,'ip12'] = trainingData.ip2.apply(lambda x: x[3]).astype(int)
	trainingData.loc[:,'ip13'] = trainingData.ip2.apply(lambda x: x[4]).astype(int)
	trainingData.loc[:,'ip14'] = trainingData.ip2.apply(lambda x: x[5]).astype(int)
	trainingData.loc[:,'ip15'] = trainingData.ip2.apply(lambda x: x[6]).astype(int)
	trainingData.loc[:,'ip16'] = trainingData.ip2.apply(lambda x: x[7]).astype(int)

	trainingData.loc[:,'ip17'] = trainingData.ip3.apply(lambda x: x[0]).astype(int)
	trainingData.loc[:,'ip18'] = trainingData.ip3.apply(lambda x: x[1]).astype(int)
	trainingData.loc[:,'ip19'] = trainingData.ip3.apply(lambda x: x[2]).astype(int)
	trainingData.loc[:,'ip20'] = trainingData.ip3.apply(lambda x: x[3]).astype(int)
	trainingData.loc[:,'ip21'] = trainingData.ip3.apply(lambda x: x[4]).astype(int)
	trainingData.loc[:,'ip22'] = trainingData.ip3.apply(lambda x: x[5]).astype(int)
	trainingData.loc[:,'ip23'] = trainingData.ip3.apply(lambda x: x[6]).astype(int)
	trainingData.loc[:,'ip24'] = trainingData.ip3.apply(lambda x: x[7]).astype(int)

	trainingData.loc[:,'ip25'] = trainingData.ip4.apply(lambda x: x[0]).astype(int)
	trainingData.loc[:,'ip26'] = trainingData.ip4.apply(lambda x: x[1]).astype(int)
	trainingData.loc[:,'ip27'] = trainingData.ip4.apply(lambda x: x[2]).astype(int)
	trainingData.loc[:,'ip28'] = trainingData.ip4.apply(lambda x: x[3]).astype(int)
	trainingData.loc[:,'ip29'] = trainingData.ip4.apply(lambda x: x[4]).astype(int)
	trainingData.loc[:,'ip30'] = trainingData.ip4.apply(lambda x: x[5]).astype(int)
	trainingData.loc[:,'ip31'] = trainingData.ip4.apply(lambda x: x[6]).astype(int)
	trainingData.loc[:,'ip32'] = trainingData.ip4.apply(lambda x: x[7]).astype(int)

	x_data = trainingData.as_matrix(columns = ['apple', 'android', 'timeStartBucket', 'weekday', 'ip01','ip02','ip03','ip04','ip05','ip06','ip07','ip08',
		'ip09','ip10','ip11','ip12','ip13','ip14','ip15','ip16','ip17','ip18','ip19','ip20','ip21','ip22','ip23','ip24',
		'ip25','ip26','ip27','ip28','ip29','ip30','ip31','ip32']).astype(np.float32)
	# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
	# y_data = trainingData.as_matrix(columns = ['duration']).astype(np.float32)
	# For Logistic Regression
	y_data = trainingData.as_matrix(columns = ['durationY'])


	print "X, Y_data initialized!!"

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.4, random_state=0)

	alphaTrScores = []
	alphaTeScores = []
	alphas = [0.0, 0.001, 0.01, 0.1, 0.2, 0.5]
	
	logitReg = linear_model.LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
	logitReg.fit(X_train, y_train)
	print logitReg.coef_
	print logitReg.get_params()

	trScore = logitReg.score(X_train, y_train)
	teScore = logitReg.score(X_test, y_test)
	print "Training set score is: ", trScore
	print "Testing set score is: ", teScore
	print "coeff 0 is: ", logitReg.coef_[0]

	plt.plot(logitReg.coef_[0], "+")
	plt.title("Logit Regression Duration Short")
	plt.savefig("images/logitRegressionShort.png")
	plt.show()

	plt.plot(logitReg.coef_[1], "+")
	plt.title("Logit Regression Duration Mid")
	plt.savefig("images/logitRegressionMid.png")
	plt.show()

	plt.plot(logitReg.coef_[2], "+")
	plt.title("Logit Regression Duration Long")
	plt.savefig("images/logitRegressionLong.png")
	plt.show()

	# for alpha in alphas:
	# 	print "Alpha is now: ", alpha
	# 	clf = linear_model.Ridge(alpha = alpha)
	# 	clf.fit(X_train, y_train)
	# 	print clf.coef_

	# 	plt.plot(clf.coef_[0], "+")
	# 	plt.title("Ridge Regression alpha = " + str(alpha))

	# 	plt.savefig("images/ridgeRegressionAlpha=" + str(alpha)+".png")
	# 	plt.show()
		
	# 	trScore = clf.score(X_train, y_train)
	# 	teScore = clf.score(X_test, y_test)
	# 	print "Training set score is: ", trScore
	# 	print "Testing set score is: ", teScore

	# 	alphaTrScores.append(trScore)
	# 	alphaTeScores.append(teScore)

	# print alphas, "\n", alphaTeScores, "\n", alphaTrScores

	# x = [0.0, 0.001, 0.01, 0.1, 0.2, 0.5] 
	# trainscores = [0.056878717619795349, 0.056878717619973429, 0.056878717621569708, 0.056878717637410814, 0.05687871765475061, 0.056878717705113657] 
	# testscores = [0.05895961652291315, 0.058959616522913261, 0.058959616522911817, 0.058959616522771374, 0.058959616522343716, 0.058959616519340452]

	# fig = plt.figure()
	# ax1 = fig.add_subplot(111)
	# ax1.scatter(x, trainscores, s=15, c='b', marker="s", label='Training')
	# ax1.scatter(x, testscores, s=15, c='r', marker="o", label='Testing')
	# plt.legend(loc = 'lower right')

	# plt.show()
