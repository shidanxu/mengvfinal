import pandas as pd
import numpy as np

data = pd.read_csv("ipToISP.csv", names = ['ip', 'city', 'state', 'isp', 'lat', 'lon', 'country'])

print data.head(50)
np.set_printoptions(threshold='nan')
print data['isp'].unique()

print data.groupby('isp').count().sort('ip')
print data.groupby('city').count().sort('ip')

# Two categories, wifi and phone network
# Wifi = Anything that contains college, university, wifi, WiFi,
wifiNetworks = ['University of Massachusetts', 'Comcast Cable', 'Verizon Internet Services', 'Charter Communications', 'Optimum Online', 'Time Warner Cable']
wirelessNetworks = ['Verizon Wireless', 'AT&T Wireless', 'Sprint PCS', 'T-Mobile USA', 'Verizon Fios']
totalNetworks = wifiNetworks + wirelessNetworks

data['wireless'] = data['isp'].isin(wirelessNetworks)
data['wifi'] = data['isp'].isin(wifiNetworks)
data['networksInfoAvailable'] = data['isp'].isin(totalNetworks)

ipToWifiWireless = data[data['networksInfoAvailable'] == True]

print ipToWifiWireless

ipToWifiWireless.to_csv("ipToWifiWireless.csv")