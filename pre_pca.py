import numpy as np
import os
import natsort
from sklearn.decomposition import PCA


files = os.listdir('../data/190305new')
file_list = natsort.natsorted(files)
print('File_list: ', file_list)

# ep = 20
# random_num = np.random.randint(low=100, high=900, size=ep)
# print('random_num:', random_num)
# data = []
#
# for i in random_num:
#     data_tmp = np.load('data/190305new/'+file_list[i])
#     data = np.hstack((data, data_tmp))  # 57210
#     # print(i, ': Done')
#     # print('data: ', data.shape)
# data = data.reshape([57210, ep]).T
# print('oriData.shape: ', data.shape)
# print('oriData: ', data)

ep = 4000
data = np.load('../data/afterSTD.npy').reshape([3600, 57210])

pca = PCA(n_components=0.95)
newData = pca.fit_transform(data).copy()
print('newData.shape:', newData.shape)
print(newData)
# reData = pca.inverse_transform(newData).copy()

np.set_printoptions(suppress=True)
np.save('../data/0_pca_3600x54_95%.npy', newData)
np.save('../data/0_pca_3600x54_95%_ratio.npy', pca.explained_variance_ratio_)
# print('newData.shape: ', newData.shape)
# print('newData: ', newData)
# print('reData.shape: ', reData.shape)
# print('reData: ', reData)
# print('D_max', np.max(reData-data), 'D-min', np.min(reData-data))
print('pca.ratio: ', pca.explained_variance_ratio_)


print('Done')
print(1)
