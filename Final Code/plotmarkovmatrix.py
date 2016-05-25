import pickle
import matplotlib.pyplot as plt
import numpy as np

# pickle.dump(distributionTest, open("dtest.pickle", "wb" ))
distributionTest = pickle.load(open("dtest.pickle", "rb"))
distributionGenerated = pickle.load(open("dgenerated.pickle", "rb"))
trainingSetNumTransitions = pickle.load(open("trainingset.pickle", "rb"))

bins = np.linspace(min(min(distributionGenerated), min(distributionTest)), max(max(distributionGenerated), max(distributionTest)), 40)
# pickle.dump(distributionGenerated, open("dgenerated.pickle", "wb" ))
# pickle.dump(trainingSetNumTransitions, open("trainingset.pickle", "wb" ))

# distributionTest.toPickle("dtest.pickle")
# distributionGenerated.toPickle("dgenerated.pickle")
# trainingSetNumTransitions.toPickle("trainingset.pickle")

plt.hist(distributionTest, bins, alpha =0.7, color= 'blue', label = 'Test Set')
# plt.hist(distributionGenerated, bins, alpha = 0.7, label = 'Markov Generated')
plt.hist(trainingSetNumTransitions, bins, alpha= 0.7, color = 'white', label = "Training Set")

plt.legend(loc='upper right')
plt.title("Distribution of Daily Number of Transitions")

plt.xlabel("Number Transitions")
plt.ylabel("Days")

plt.savefig("Markov On All Recolored Test Training.png")

plt.show()