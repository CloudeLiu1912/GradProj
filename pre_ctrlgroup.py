import numpy as np
import natsort
import os
import time
import matplotlib.pyplot as plt

start_time = time.time()
files = os.listdir('data/190305')
file_list = natsort.natsorted(files)
print('filelist: ', file_list)

x = np.load('spa_lines.npy')


def get_ctrlgroup(key=True):
    if key:
        data = []
        for n in range(1):
            for i in range(100):
                data_ = np.loadtxt('data/190305/'+file_list[100*n+i], skiprows=1, usecols=1)
                # print('data_ =', data_)
                print('file =', file_list[100*n+i])
                data_tmp = (data_ - np.mean(data_, axis=0)) / np.std(data_, axis=0)
                # print('data_tmp =', data_tmp)
                np.save('data/190305new/'+str(n)+'_'+str(i+1)+'.npy', data_tmp)
                data.append(data_tmp)
        np.save('data/afterSTD_ctrlgroup.npy', data)
    else:
        data = np.load('data/afterSTD_ctrlgroup.npy')
        pass
    return data


ctrl_group = get_ctrlgroup(key=False)
aver = np.average(ctrl_group, axis=0)
print(ctrl_group.shape)
print(aver.shape)
print(aver)

td = np.load('data/190305new/1_1.npy')
a = td - aver
print('max:', np.max(a), 'min:', np.min(a))

plt.figure()
plt.plot(x, a, x, td)
plt.show()

end_time = time.time()
print('Total time:', end_time-start_time)
print(1)
