#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import os
from datetime import datetime

ipDict = {}
allDevices = {}
devices = []
def ipFeature(line):
    try:
        start, end, ip, device, identity = line.split(";")
    except ValueError:
        return None

    ip = ip.strip()
    ip1, ip2, ip3, ip4 = [int(x) for x in ip.split(".")]
    vec = "0"* 32
    first8 = '{0:08b}'.format(ip1)
    second8 = '{0:08b}'.format(ip2)
    third8 = '{0:08b}'.format(ip3)
    fourth8 = '{0:08b}'.format(ip4)

    vec = first8 + second8 + third8 + fourth8

    # print vec
    # print list(vec)
    return [int(x) for x in list(vec)]

def timeStartFeature(line):
    try:
        start, end, ip, device, identity = line.split(";")
        hour, minute, sec = [int(x) for x in start.strip().split(":")]
    except ValueError:
        return None
        
    first5 = '{0:05b}'.format(hour)
    second6 = '{0:06b}'.format(minute)
    third6 = '{0:06b}'.format(sec)


    vect = first5 + second6 + third6

    # print vect
    return [int(x) for x in list(vect)]

def timeEndFeature(line):
    start, end, ip, device, identity = line.split(";")
    hour, minute, sec = [int(x) for x in end.strip().split(":")]

    first5 = '{0:05b}'.format(hour)
    second6 = '{0:06b}'.format(minute)
    third6 = '{0:06b}'.format(sec)


    vect = "0" * 17
    vect[0:5] = first5
    vect[5:11] = second6
    vect[11:17] = third6

    print vect
    return list(vect)

def device(line):
    start, end, ip, device, identity = line.split(";")
    device = device.strip()

    devices.append(device)
    
    # This is the index of device
    if device in allDevices:
        return allDevices[device]
    else:
        allDevices[device] = len(devices) - 1
        return allDevices[device]
    # clean data


def duration(line):
    try:
        start, end, ip, device, identity = line.split(";")
        FMT = '%H:%M:%S'
        tdelta = datetime.strptime(end.strip(), FMT) - datetime.strptime(start.strip(), FMT)    
    except ValueError:
        return None
    return tdelta.seconds

# What the start hour is
def timeOfDay(line):
    try:
        start, end, ip, device, identity = line.split(";")
        hour, minute, sec = [int(x) for x in start.strip().split(":")]
    except ValueError:
        return None
        
    # first5 = '{0:05b}'.format(hour)
    # second6 = '{0:06b}'.format(minute)
    # third6 = '{0:06b}'.format(sec)

    return hour

def dayOfWeek(line):

def durationLessThanMinute(line):
    return duration(line)/60 < 1

def durationOneToFive(line):
    minutes = duration(line)/60
    return minutes > 0 and minutes < 5

def durationFiveOrMore(line):
    minutes = duration(line)/60
    return minutes >= 5

# midnight cases
# construct more data for time
