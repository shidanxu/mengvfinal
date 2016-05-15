import numpy as np
import urllib2
import json
import time
import csv


filename = "ipToISP.csv"

uniqueIPArray = np.load("uniqueIPs.npy")

print uniqueIPArray, len(uniqueIPArray)

count = 0
startIndex = 111239
for counter in range(len(uniqueIPArray)):
	count += 1
	if counter < startIndex:
		continue
	# Look up and append to file
	# 150 per minute
	# Do 140, wait 60 sec
	ip = uniqueIPArray[counter]
	print ip
	data = json.load(urllib2.urlopen("http://ip-api.com/json/" + ip))
	print "data is: ", data
	if data['status'] == 'success':
		arr = [ip, data['city'], data['region'], data['isp'], data['lat'], data['lon'], data['country']]
		with open(filename, "a") as output:
			writer = csv.writer(output, lineterminator='\n')
			try:
				writer.writerow(arr)
			except UnicodeEncodeError:
				continue

	if count % 120 == 0:
		print "Sleeping..."
		time.sleep(60)

	if count % 100 == 0:
		print "\n\n\n" + str(count) + " out of " + str(len(uniqueIPArray)) + " finished.\n\n\n"

