import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sklearn

# agg = lambda x: (x['Timestamp'].astype('i8') * (x['Value'].astype('f8') / x['Value'].sum())).sum()

entireData = pd.read_pickle("../../Large Data/datasetMayDur.pickle")
# entireData['duration'] = (entireData['timeEnd'] - entireData['timeStart']).astype('timedelta64[s]')
# entireData.to_pickle("../../Large Data/datasetMayDur.pickle")

entireData["timeStart"] = (entireData["timeStart"].astype('i8') / 1000000000) % 86400
entireData["timeEnd"] = (entireData["timeEnd"].astype('i8') / 1000000000) % 86400
print entireData.head(20)

print type(entireData.groupby("id").count()['timeStart'])
# afkldsjjflasdjk

grouped = entireData.groupby("id").mean()
grouped = grouped[grouped['duration'] > 0]
grouped['duration'].hist(bins = 100)

grouped['numberSessions'] = entireData.groupby(["id", "date"]).count()['timeStart'].reset_index().groupby("id").mean()


print grouped.head(20)

wifi = grouped[grouped['wifi'] == True]
wireless = grouped[grouped['wireless'] == True]

grouped['duration'].hist(bins = 100)
plt.show()

# Here we have startTimeAvgByUser
# It is normally distributed for several reasons.
# 1. We cannot separate the periodic sessions of ipads.
# So the data is predominantly useless
# 2. A suggestion is to take out the users of larger than n sessions per day.
# 3. No canonical user is found other than the normally distributed mid at 12.
grouped.timeStart = (grouped.timeStart / 3600)
grouped.timeEnd = (grouped.timeEnd / 3600)
grouped.timeStart.hist(bins = 24)
plt.show()

grouped[grouped.numberSessions < grouped.numberSessions.quantile(.98)]['numberSessions'].hist(bins = 100)

plt.xlabel("Average Number of Sessions Per Day")
plt.ylabel("Number of users")
plt.title("Users by Mean Number of Daily Sessions")
plt.savefig("../images/Users by Mean Number of Daily Sessions.png")
plt.show()

# Plot individual distributions
grouped = grouped[grouped.numberSessions < grouped.numberSessions.quantile(.98)]
grouped.hist(bins = 40)
plt.savefig("../images/Mean Feature Values per User HIST.png")
plt.show()

# Then goes to kmeans.py
grouped.to_pickle("../datasets/groupedByUserAll.pickle")