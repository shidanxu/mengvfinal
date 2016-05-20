import random, string
import os
import time
import glob
from datetime import datetime

markov = [[0.99, 0.01, 0], [0.1, 0.8, 0.1], [0, 0.05, 0.95]]

def generateEntry(startDate, endDate, md5, ips, path):
	# ip = generateIp()	
	# md5 = generateRandomStringLength(60)

	for i in range(int(random.random()*10) + 1):
		ip = random.choice(ips)
		time1 = generateTime()
		time2 = generateTime()
		time1, time2 = compareTwoTimes(time1, time2)

		with open(os.path.join(path, md5+".csv"), 'a') as f:
			line = ','.join([time1, time2, ip, 'iPhone', md5])
			print line
			f.write(line + "\n")


def generateIp():
	value1 = int(random.random()*255)
	value2 = int(random.random()*255)
	value3 = int(random.random()*255)
	value4 = int(random.random()*255)
	
	return str(value1) + "." + str(value2) + "." + str(value3) + "." + str(value4)

def generateTime():
	hour = int(random.random()*24)
	minute = int(random.random()*60)
	second = int(random.random()*60)

	strHour = ("0" + str(hour))[-2:]
	strMinute = ("0" + str(minute))[-2:]
	strSecond = ("0" + str(second))[-2:]
	return strHour + ":" + strMinute + ":" + strSecond



def generateRandomStringLength(X):
	return ''.join(random.choice(string.lowercase + string.digits) for x in range(X))

def compareTwoTimes(time1, time2):
	if int(time1[:2]) < int(time2[:2]): return (time1, time2)
	if int(time1[:2]) > int(time2[:2]): return (time2, time1)
	if int(time1[3:5]) < int(time2[3:5]): return (time1, time2)
	if int(time1[3:5]) > int(time2[3:5]): return (time2, time1)
	if int(time1[6:8]) < int(time2[6:8]): return (time1, time2)
	if int(time1[6:8]) > int(time2[6:8]): return (time2, time1)
	return (time1, time2)


def parseEntry(path, filename):
	startingState = 1
	entries = []
	with open(os.path.join(path, filename), 'r') as f:
		for line in f:
			# timeStart, timeEnd, ip, device, id = [item.strip() for item in line.split(",")]
			try:
				timeStart, timeEnd, ip, device, id = [item.strip() for item in line.split(";")]
				timeStart = datetime.strptime(timeStart, "%H:%M:%S")
				timeEnd = datetime.strptime(timeEnd, "%H:%M:%S")
			except ValueError:
				return

			entries.append(Entry(timeStart, timeEnd, ip, device, id))


	combinedActions = []
	for entry in entries:
		combinedActions.append((entry.ip, "join", entry.startTime))
		combinedActions.append((entry.ip, "leave", entry.endTime))

	combinedActions = sorted(combinedActions, key = lambda x: x[2])
	# print combinedActions

	# State 1 => not online
	# State 2 => Online with 1 device
	# State 3 => Online wiht >=2 devices
	states = [(datetime.strptime("00:00:00", "%H:%M:%S"), 1, [])]
	devicesOnline = set()

	for (ip, action, time) in combinedActions:
		# print (ip, action, time)
		currentState = states[-1][-1]
		if action == "join":
			devicesOnline.add(ip)
			if len(devicesOnline) == 1:
				states.append((time, 2, list(devicesOnline)))
			elif len(devicesOnline) > 1:
				states.append((time, 3, list(devicesOnline)))
			else:
				states.append((time, 1, list(devicesOnline)))
		if action == "leave":
			# This line will not work in the current random dataset
			# assert ip in devicesOnline
			if ip in devicesOnline:
				devicesOnline.remove(ip)
			if len(devicesOnline) == 1:
				states.append((time, 2, list(devicesOnline)))
			elif len(devicesOnline) > 1:
				states.append((time, 3, list(devicesOnline)))
			else:
				states.append((time, 1, list(devicesOnline)))

	# Make sure everyone logged off
	# assert len(devicesOnline) == 0
	return states

			

		 


class Entry:
	def __init__(self, startTime, endTime, ip, device, id):
		self.startTime = startTime
		self.endTime = endTime
		self.ip = ip
		self.device = device
		self.id = id



# print(generateIp())
# print generateTime()
# print string.lowercase
# print generateRandomStringLength(60)

# for i in range(100):
# 	generateEntry(1, 2, generateRandomStringLength(60), [generateIp() for x in range(3)], "fakeData")
