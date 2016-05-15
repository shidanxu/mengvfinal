import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle


if __name__ == '__main__':
	# basepath = "pickleData/"

	# entireData = pd.DataFrame()
	# for files in os.listdir(basepath):
	# 	pandasFile = pd.read_pickle(basepath + files)

	# 	print pandasFile
	# 	entireData = pd.concat([entireData, pandasFile])


	# print "Entire Dataset: ", entireData
	# entireData.to_pickle("allJoinedData.pickle")

	entireData = pd.read_pickle("ipython/processedWithDates.pickle")
	entireData['apple'] = (entireData['device'].isin(['I-phone/I-pad', 'iPhone', 'iPad', 'Mac', 'iPod'])).astype(int)
	entireData['android'] = (entireData['device'] == 'android').astype(int)
	entireData['duration'] =  (entireData.timeEnd-entireData.timeStart).astype('timedelta64[s]')
	entireData['timeStartBucket'] = entireData.timeStart.apply(lambda x: x.hour)


	print entireData.head(50)
	print entireData['device'].unique()
	print entireData['id'].head()
	print entireData['ip'].head()
	print entireData['date'].head()
	print entireData['timeStartBucket'].unique()
