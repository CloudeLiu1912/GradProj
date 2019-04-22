import numpy as np
import pandas as pd
import os
import natsort
import matplotlib.pyplot as plt
import pandas as pd


# print(int(file[0:2]))
# for roots, dirs, files in os.walk('20190119'):
#     print('roots=', roots)
#     print('dirs=', dirs)
#     print('files=', files)
#     for file in files:
#         data_tmp = np.loadtxt(file, skiprows=1)


# x = np.load('spa_lines.npy')
# y = np.load('1_002.spa.npy')

# plt.plot(x, y)
# plt.show()
# print(1)


# t0 = np.loadtxt('LIBS OpenData csv/csv Certified Samples Subset 1000pulseaverage/Andesite73302_1000AVG.csv',
#                 delimiter=',', usecols=[1])
# t1 = np.load('LIBS_data/Andesite73302_1000AVG.csv.npy')
# t2 = pd.read_csv('LIBS OpenData csv/csv Certified Samples Subset 1000pulseaverage/Andesite73302_1000AVG.csv',
#                  header=None, usecols=[1]).values.flatten()
# tt = []
# tt.append(t0)
# tt.append(t1)
#
# np.array(tt)
#
# x = np.load('LIBS_data/libs_lines.npy')
#
# print('t0.type', t0)
# print('t1.type', t1)
# print('t2.type', t2)
# print('tt.type', tt)


# def get_local_min(x, t=0):
#     min1 = t-1
#     max1 = t+1
#     for a in x:
#         if (a >= min1) and (a <= max1):
#             index = np.argwhere(x == a)[0][0]
#             print(a, '::', index)
#         pass
#
#     return 0
#
#
# def get_wavelet_cal(its, spa, f_line, eps=1.0):
#     min_ = f_line-eps
#     max_ = f_line+eps
#     sum_ = 0
#     for a in spa:
#         if (a >= min_) and (a <= max_):
#             index_ = np.argwhere(spa == a)[0][0]
#             sum_ += its[index_]  # [index][1]:the spectral intensity
#     return sum_


# x = np.loadtxt('data/190305/0_001.spa', delimiter=' ', skiprows=1, usecols=[0])  # lines / spa
# y = np.loadtxt('data/190305/0_001.spa', delimiter=' ', skiprows=1, usecols=[1])  # data / intensity
# z = np.loadtxt('data/190305/0_001.spa', delimiter=' ', skiprows=1)
#
# print('x=', x)
# print('y=', y)
# print('z=', z[0][1])
#
# print('sum = ', np.sum(x[0:2]))
# #
# s = get_wavelet_cal(y, x, 307.562, 0.001)
# print('s=', s)

# plt.plot(x, y)
# plt.show()
# get_local_min(x, t=300)


# a = np.array(([3,2,1],[2,5,7],[4,7,8]))
# itemindex = np.argwhere(a == 7)
# print(itemindex)

# x = np.arange(12).reshape([3, 4])
# y = np.arange(3).reshape([3, 1])
#
# print('x:', x)
# print('y:', y)
#
# xx = x[x[:, 0] <= 2].copy
# y = y[x[:, 0] <= 4]
# # print('x[0, :] ::', x[:, 0] <= 4)
# print('x:', x)
# print('y:', y)

x = np.load('data/new_features_144x4000.npy')
x = x.reshape([4000, 144])
# print(x)
np.savetxt('data/new_features_144x4000.csv', x, delimiter=',')

print('Done!')
