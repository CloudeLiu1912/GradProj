import numpy as np
import pandas as pd
import natsort
import matplotlib.pyplot as plt
import os


dirt = 'LIBS OpenData csv/csv Certified Samples Subset 1000pulseaverage'


def get_label(key=True):
    if key:
        # label = np.loadtxt('LIBS OpenData csv/Sample_Composition_Data.csv', delimiter=',',
        #                    skiprows=1, usecols=[4], encoding='utf-8')
        label = pd.read_csv('LIBS OpenData csv/Sample_Composition_Data.csv', delimiter=',',
                            usecols=[3], encoding='utf-8').dropna().replace('-', 0)
        label = label.values
        np.save('LIBS_data/0_FeO_label.npy', label)
        ret = label
    else:
        ret = np.load('LIBS_data/0_FeO_label.npy')
        pass
    return ret


def get_spa_lines(key=True):
    if key:
        ret = np.loadtxt('LIBS OpenData csv/csv Certified Samples Subset 1000pulseaverage/Andesite73302_1000AVG.csv',
                         usecols=[0], delimiter=',')
        np.save('LIBS_data/0_libs_lines.npy', ret)
    else:
        ret = np.load('LIBS_data/0_libs_lines.npy')
        pass
    return ret


def get_data(key=True):
    ret = 0
    all_data = []
    if key:
        files = os.listdir(dirt)
        files = natsort.natsorted(files)
        for file in files:
            d_tmp = np.loadtxt(dirt+'/'+file, usecols=[1], delimiter=',')
            np.save('LIBS_data/'+file+'.npy', d_tmp)
            all_data.append(d_tmp)
        ret = all_data
        np.save('LIBS_data/0_all_data.npy', ret)
    else:
        files = os.listdir('LIBS_data')
        files = natsort.natsorted(files)
        ret = np.load('LIBS_data/0_all_data.npy')
        pass

    return ret


# lines = get_spa_lines(key=False)
# tmpData = get_data(tmp=0)

print('label =', get_label(key=False).shape)
print('Lines =', get_spa_lines(key=False).shape)
print('data =', get_data(key=False).shape)
print(1)
