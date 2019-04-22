import numpy as np
import natsort
import time
import matplotlib.pyplot as plt
import pandas as pd


start_time = time.time()

ctrl_group = np.load('data/ctrl_group.npy')
x = np.load('data/spa_lines.npy')
y = np.loadtxt('data/190305/1_001.spa', skiprows=1, usecols=1)

print(y)

plt.plot(x, y-ctrl_group)
plt.show()

r = np.vstack((x, y-ctrl_group))
print('r:', r)
np.savetxt('data/0_1_1.txt', r, delimiter=',')

print('ctrl group:', ctrl_group.shape)
print(ctrl_group)

end_time = time.time()
print('Total time: ', end_time-start_time)
print(1)
