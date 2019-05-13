import numpy as np
from sklearn.decomposition import PCA
import time

start_time = time.time()

soil = np.load('../data/afterSTD_soil.npy').reshape(400, 57210)
data = np.load('../data/afterSTD.npy').reshape(3600, 57210)
print('data:', data.shape)
print('soil:', soil.shape)
# print(soil)

pca = PCA(n_components=0.9)
newsoil = pca.fit_transform(soil)
newdata = pca.transform(data)
print('new data:', newdata.shape)
print(newdata)

# x = np.arange(400*55).reshape([400, 55])
# y = pca.inverse_transform(x)
# print('x:', x)
# print('y:', y.shape)
#
# xx = np.arange(1000*57210).reshape(1000, 57210)
# a = pca.transform(xx)
# print('a:', a.shape)
# print(a)

np.save('../data/soil_pca_soil_400x88.npy', newsoil)
np.save('../data/soil_pca_plant_3600x88.npy', newdata)

end_time = time.time()
print('Total time: ', end_time-start_time)
print(1)
