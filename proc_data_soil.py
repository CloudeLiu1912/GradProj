import numpy as np
import natsort
import os
import time
from sklearn.decomposition import PCA

start_time = time.time()

files = os.listdir('data/190305')
file_list = natsort.natsorted(files)
print('filelist: ', file_list)

soil = []
for nub in range(3700, 4100):  # 4100
    print(file_list[nub])
    d_ = np.loadtxt('data/190305/'+file_list[nub], skiprows=1, usecols=1)
    soil = np.hstack((soil, d_))
    pass

r = soil.reshape(int(len(soil)/57210), 57210)

print('soil:', r.shape)
print(r)
np.save('data/row_soil.npy', r)

# data = np.load('data/40_100.spa.npy')
# print('data:', data.shape)
# print(data)

print('Done!')
print(1)
