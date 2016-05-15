# This code predicts the length of each duration using 4 features,
# and outputs one of the 3 tags 'short', 'mid', 'long' for length of duration
# Using a one layer neural net with 10 neurons, this achieves a 95.4% accuracy
# classifying the duration tags. Mainly because the tags are concentrated towards short.

# This is just a demonstration of a benchmark for using neural net approach.

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn import cross_validation
import input_data
import datetime

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

def durationY(duration):
	if duration < 5:
		return 'short'
	elif duration < 3600:
		return 'mid'
	else:
		return 'long'
	
	

# mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# print "TRX: ", trX, trX.shape
# print "\n"
# print "TRY: ", trY, trY.shape
# print "\n"
# print "TEX: ", teX, teX.shape
# print "\n"
# print "TEY: ", teY, teY.shape

entireData = pd.read_pickle("ipython/training_50000.pickle")

trainingData = entireData.head(50000)
# trainingData.to_pickle("ipython/training_50000.pickle")
trainingData.loc[:, 'apple'] = (trainingData['device'].isin(['I-phone/I-pad', 'iPhone', 'iPad', 'Mac', 'iPod'])).astype(int)
trainingData.loc[:, 'android'] = (trainingData['device'] == 'android').astype(int)
trainingData.loc[:, 'duration'] =  (trainingData.timeEnd-trainingData.timeStart).astype('timedelta64[s]')
trainingData.loc[:, 'durationShort'] = (trainingData.duration < 5).astype(int)
trainingData.loc[:, 'durationLong'] = (trainingData.duration >=3600).astype(int)
trainingData.loc[:, 'durationMid'] = np.invert(np.logical_or(trainingData.durationShort, trainingData.durationLong)).astype(int)

trainingData.loc[:, 'timeStartBucket'] = trainingData.timeStart.apply(lambda x: x.hour)
trainingData.loc[:, 'durationY'] = trainingData.duration.apply(durationY)

x_data = trainingData.as_matrix(columns = ['apple', 'android', 'timeStartBucket', 'weekday']).astype(np.float32)
y_data = trainingData.as_matrix(columns = ['durationShort', 'durationMid', 'durationLong']).astype(np.float32)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.4, random_state=0)
# X_train = X_train.transpose()
# X_test = X_test.transpose()

print X_test
print y_test
# This is the breakpoint

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])

w_h = init_weights([4, 10]) # create symbolic variables
w_o = init_weights([10, 3])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.03).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


for i in range(100):
    for start, end in zip(range(0, len(X_train), 128), range(128, len(X_train), 128)):
    	# print start, end
        sess.run(train_op, feed_dict={X: X_train[start:end], Y: y_train[start:end]})
        # time_str = datetime.datetime.now().isoformat()
        # print result

    print i, np.mean(np.argmax(y_test, axis=1) ==
                     sess.run(predict_op, feed_dict={X: X_test, Y: y_test}))