import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import natsort
from sklearn.decomposition import PCA


data = np.loadtxt('data/190305/1_001.spa', skiprows=1)
# d = data[data[]]
print('data:', data.shape, ' \n', data)
d = data[data[:, 1] >= 300]
print('d:', d.shape)

print('Done')
print(1)
