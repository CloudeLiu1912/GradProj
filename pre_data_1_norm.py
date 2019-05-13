import numpy as np
import os
import natsort

c = [4.021, 3.608, 3.986, 1.491, 3.618, 3.046, 3.535, 3.178,
     8.664, 8.481, 8.482, 7.598, 8.971, 8.973, 8.226, 8.386, 9.303, 9.306,
     6.356, 6.132, 7.806, 8.324, 8.267, 7.941, 8.927, 7.486,
     5.707, 4.946, 9.612, 8.944, 6.528, 6.772, 7.593, 7.238, 7.621, 8.494,
     6.656, 4.576, 5.585, 4.826]

files = os.listdir('../data/190305')
file_list = natsort.natsorted(files)
print('filelist: ', file_list)

x = np.load('../data/spa_lines.npy')

data = []
for n in range(1, 37):
    for i in range(100):
        data_ = np.loadtxt('../data/190305/'+file_list[100*n+i], skiprows=1, usecols=1) / c[n-1]  # +100
        # print('data_ =', data_)
        print('file =', file_list[100*n+i])
        data_tmp = (data_ - np.mean(data_, axis=0)) / np.std(data_, axis=0)
        # print('data_tmp =', data_tmp)
        # np.save('data/190305new/'+str(n+1)+'_'+str(i+1)+'.npy', data_tmp)
        data.append(data_tmp)
    pass

# plt.plot(x, data[0])
# plt.show()
data = np.array(data)

print('data = ', data.shape)
np.save('../data/afterSTD.npy', data)
print(1)
