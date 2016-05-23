# 1. Parse a file to get a list of states [0, 1, 2, 2, 2, 1, ... ,0]

# 2. Parse all files to generate the markov matrix transition probabilities

# 3. Generate data from the Markov matrix

# 4. Compare and contrast results
import random_generator
import numpy as np
import os
from sklearn.preprocessing import normalize
import random
import matplotlib.pyplot as plt
import features
import pandas as pd

def get_all_files(mypath):
	return [ f for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,f)) ]



def statesToPeriod(states, timePeriod = 20):
	lengthPeriods = 24*60 / timePeriod
	# print lengthPeriods
	periods = [1]*lengthPeriods
	

	currentState = 1
	currentStateStartIndex = 0
	for eachTuple in states:
		time, state, ip = eachTuple
		tillTimeIndex = timeToIndex(time)
		# for ind in range(currentStateStartIndex, tillTimeIndex):
			# periods[ind] = currentState

		periods[tillTimeIndex] = state
		currentState = state
		currenStateStartIndex = tillTimeIndex
	
	# print periods
	return periods

def timeToIndex(datetimeObj, timePeriod = 20):
	assert 60 / timePeriod
	return datetimeObj.hour * 60 / timePeriod + datetimeObj.minute / timePeriod

def computeTransitionMatrix(periods, states = 3):
	totalNumTransitions = 0
	transitionMatrix = []
	for jj in range(states):
		transitionMatrix.append([0]*states)
	# print transitionMatrix

	for i in range(len(periods) - 1):
		begin = periods[i] - 1
		end = periods[i+1] - 1
		# print begin, end

		transitionMatrix[begin][end] += 1
		if begin != end:
			totalNumTransitions += 1

		# print transitionMatrix
	assert sum([sum(item) for item in transitionMatrix]) == len(periods) - 1

	# print "the matrix: ", np.matrix(transitionMatrix)
	return np.matrix(transitionMatrix)

def computeProbabilityMatrix(transitionMatrix):
	# print "to be normalized: ", transitionMatrix
	normed_matrix = normalize(transitionMatrix.astype(float), axis=1, norm='l1')
	# print normed_matrix
	return normed_matrix

def generateDataFromMarkovMatrix(markovMatrix, period = 20):
	numStates = markovMatrix.shape[0]
	# print numStates
	sampleLength = 60*24 / period

	output = [1]
	currentState = 1
	# Run 1000 rounds to get current state some not always 1 state
	for j in range(100):
		randomNum = random.random()
		# print "the random number is:", randomNum
		if currentState == 1:
			cumProb = np.cumsum(markovMatrix[0])
			# print "state = 0, cumprob:", cumProb
			for jj in range(len(cumProb)):
				if randomNum < cumProb[jj]:
					currentState = jj + 1
					# print "i, currentState:", i, currentState
					break
		elif currentState == 2:
			cumProb = np.cumsum(markovMatrix[1])
			# print "state = 1, cumprob:", cumProb
			for jj in range(len(cumProb)):
				if randomNum < cumProb[jj]:
					currentState = jj + 1
					# print "i, currentState:", i, currentState
					break
		elif currentState == 3:
			cumProb = np.cumsum(markovMatrix[2])
			# print "state = 2, cumprob:", cumProb
			for jj in range(len(cumProb)):
				if randomNum < cumProb[jj]:
					currentState = jj + 1
					# print "i, currentState:", i, currentState
					break
		else:
			print "ERROR: Current State is: " + str(currentState)

	for i in range(sampleLength - 1):
		# raw_input()
		randomNum = random.random()
		# print "the random number is:", randomNum
		if currentState == 1:
			cumProb = np.cumsum(markovMatrix[0])
			# print "state = 0, cumprob:", cumProb
			for jj in range(len(cumProb)):
				if randomNum < cumProb[jj]:
					output.append(jj + 1)
					currentState = jj + 1
					# print "i, currentState:", i, currentState
					break
		elif currentState == 2:
			cumProb = np.cumsum(markovMatrix[1])
			# print "state = 1, cumprob:", cumProb
			for jj in range(len(cumProb)):
				if randomNum < cumProb[jj]:
					output.append(jj + 1)
					currentState = jj + 1
					# print "i, currentState:", i, currentState
					break
		elif currentState == 3:
			cumProb = np.cumsum(markovMatrix[2])
			# print "state = 2, cumprob:", cumProb
			for jj in range(len(cumProb)):
				if randomNum < cumProb[jj]:
					output.append(jj + 1)
					currentState = jj + 1
					# print "i, currentState:", i, currentState
					break
		else:
			print "ERROR: Current State is: " + str(currentState)

		# print "output: ", output
	# print output, len(output)
	return output
# Evaluate 1 compares the distribution of number of transitions each day for both the generated and testing actual data
def evaluate1(trainingSetNumTransitions, dailyStates, size = 10000, basepath = '../../../alllogs/'):
	print "length of dailyStates,", len(dailyStates)
	print "size: ", size
	assert len(dailyStates) == size
	assert len(trainingSetNumTransitions) == size
	limit = size

	distributionGenerated = []
	distributionTest = []
	done = False

	for dayStates in dailyStates:
		distributionGenerated.append(countTransitions(dayStates))


	for files in os.listdir(basepath):
		if done == True:
			break	
		path = os.path.join(basepath, files)
		
		if os.path.isdir(path):
			for logFile in os.listdir(path):
				with open(os.path.join(path, logFile), 'r') as f:				
					states = random_generator.parseEntry(path, logFile)
					
					if states != None:
						limit -= 1
						periodStates = statesToPeriod(states)
						distributionTest.append(countTransitions(periodStates))

				if limit == 0:
					done = True
					break

	# Plot
	# plt.subplot
	print "distributionGenerated: "
	print distributionGenerated
	print "\n distributionTest: "
	print distributionTest
	# plt.hist(distributionTest)
	# plt.hist(distributionGenerated)
	# plt.savefig("results.png")

	print len(distributionGenerated)
	print len(distributionTest)

	bins = np.linspace(min(min(distributionGenerated), min(distributionTest)), max(max(distributionGenerated), max(distributionTest)), 50)
	distributionTest.toPickle("dtest.pickle")
	distributionGenerated.toPickle("dgenerated.pickle")
	trainingSetNumTransitions.toPickle("trainingset.pickle")

	plt.hist(distributionTest, bins, alpha =0.7, color = 'green', label = 'Test Set')
	plt.hist(distributionGenerated, bins, alpha = 0.7, color= 'black',label = 'Markov Generated')
	plt.hist(trainingSetNumTransitions, bins, alpha= 0.7, color='blue',label = "Training Set")

	plt.legend(loc='upper right')
	plt.title("Distribution of Daily Number of Transitions")

	plt.xlabel("Number Transitions")
	plt.ylabel("Days")

	plt.savefig("Markov On All.png")

	plt.show()
	return

def countTransitions(dayStates):
	total = 0
	currentState = dayStates[0]
	for state in dayStates:
		if state != currentState:
			total += 1
			currentState = state
	return total

def doMarkovNaive(testSampleSize = 10000):
	idToCluster = pd.read_pickle("../datasets/mergedWithCluster.pickle")

	totalTransitionMatrix = np.matrix([[0,0,0], [0,0,0], [0,0,0]])
	cluster1Matrix = np.matrix([[0,0,0], [0,0,0], [0,0,0]])
	cluster2Matrix = np.matrix([[0,0,0], [0,0,0], [0,0,0]])
	cluster3Matrix = np.matrix([[0,0,0], [0,0,0], [0,0,0]])
	

	limit = testSampleSize
	finished = False
	parse = True
	basepath = '../../../alllogs/'

	trainingSetNumTransitions = []


	try:
		os.listdir(basepath)
	except Exception, e:
		print e
		parse = False
	

	if parse:
		# File name is id, can separate by cluster
		# files in basepath are just dates
		for files in os.listdir(basepath):
			if finished == True:
				break
			path = os.path.join(basepath, files)
			if os.path.isdir(path):
				for logFile in os.listdir(path):
					# logFile contains ID
					print logFile
					try:
						cluster = int(idToCluster.loc[logFile]['cluster'])
					except KeyError:
						continue
					
					print "Cluster for " + logFile + " is " + str(cluster) + "\n"

					if limit == 0:
						finished = True
						break

					states = random_generator.parseEntry(path, logFile)
					if states != None:

					# print states

						periods = statesToPeriod(states)
						transitionMatrix = computeTransitionMatrix(periods)
						numTrans = countTransitions(periods)
						trainingSetNumTransitions.append(numTrans)

						totalTransitionMatrix = totalTransitionMatrix + transitionMatrix
						if cluster == 1:
							cluster2Matrix += transitionMatrix
						if cluster == 0:
							cluster1Matrix += transitionMatrix
						if cluster == 2:
							cluster3Matrix += transitionMatrix

						limit -= 1
					# print limit
					# print totalTransitionMatrix

	# print "Training finished."
	# print totalTransitionMatrix
	normed_matrix = computeProbabilityMatrix(totalTransitionMatrix)
	normed_c1 = computeProbabilityMatrix(cluster1Matrix)
	normed_c2 = computeProbabilityMatrix(cluster2Matrix)
	normed_c3 = computeProbabilityMatrix(cluster3Matrix)
	
	print "\nNormed Matrices:"
	print normed_matrix

	print "\nNormed C1"
	print normed_c1

	print "\nNormed C2"
	print normed_c2

	print "\nNormed C3"
	print normed_c3
	

	testTransitionMatrix = np.matrix([[0,0,0], [0,0,0], [0,0,0]])
	dailyStates = []

	# print "\n\n"
	for kk in range(testSampleSize):
		markov_generated = generateDataFromMarkovMatrix(normed_matrix)
		dailyStates.append(markov_generated)

		transitionMatrix = computeTransitionMatrix(markov_generated)
		# print transitionMatrix
		# trainingSetNumTransitions.append(numTrans)

		testTransitionMatrix = testTransitionMatrix + transitionMatrix

	# print "Testing data generated."
	# print testTransitionMatrix

	normed_matrix_test = computeProbabilityMatrix(testTransitionMatrix)
	# print normed_matrix_test

	evaluate1(trainingSetNumTransitions, dailyStates, size = testSampleSize)


def featureAvg():
	currentPath = os.getcwd() + "../../"
	oldpath = "../../alllogs"

	files = get_all_files(oldpath)

	for afile in files:
		print afile
		with open(os.path.join(currentPath, afile), 'r') as f:
			content = f.readlines()
			count = 0
			for line in content:
				count += 1
			print count
		
	totalTransitionMatrix = np.matrix([[0,0,0], [0,0,0], [0,0,0]])
	# for files in os.listdir("fakeData"):
	testSampleSize = 30

	limit = 30
	finished = False
	parse = True
	basepath = '../../alllogs/'

	try:
		os.listdir(basepath)
	except Exception, e:
		print e
		parse = False
	
	arrayOfFeatures = []
	if parse:
		for files in os.listdir(basepath):
			if finished == True:
				break
			path = os.path.join(basepath, files)
			if os.path.isdir(path):
				for logFile in os.listdir(path):
					if limit == 0:
						finished = True
						break

					with open(os.path.join(path,logFile), 'r') as h:					
						lines = h.readlines()
						for line in lines:
							if line:
								feature = [features.durationLessThanMinute(line)]
								feature.extend([features.durationOneToFive(line)])
								feature.extend([features.durationFiveOrMore(line)])
								feature.extend([features.device(line)])
								feature.extend(features.ipFeature(line))
								feature.extend(features.timeStartFeature(line))
								feature.extend(features.timeEndFeature(line))

								arrayOfFeatures.append(feature)

					
					limit -= 1
					print "limit: ", limit

	nparray = np.array(arrayOfFeatures)
	print nparray.shape
	row, col = nparray.shape

	plotted = np.divide(np.sum(nparray, axis=0), float(row))
	print plotted

	import matplotlib.pyplot as plt
	plt.plot(plotted)
	plt.ylabel('Distribtuton')
	plt.savefig("1.png")

	


if __name__ == '__main__':
	doMarkovNaive()

	