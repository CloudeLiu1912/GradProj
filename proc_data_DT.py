import numpy as np
import os
import time
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

start_time = time.time()

# 1: Row data
# all_data = np.load('../data/pca_4000x6.npy')
# print('all.shape', all_data.shape)
# re_all_data = all_data.reshape([4000, 6])
# row_label = np.load('../data/label.npy')
# # 2: Slide
# blank_ = re_all_data[:100].copy()
# x_ = re_all_data[100:3700].copy()  # ginseng
# soil_ = re_all_data[3700:].copy()  # soil
# y_ = row_label[:3600].reshape([3600, 1]).copy()
# print('blank:', blank_.shape)
# print('x:', x_.shape)
# print('y:', y_.shape)

x_ = np.load('../data/soil_pca_plant_3600x88.npy')
y_ = np.load('../data/label.npy')[:3600].reshape([3600, 1])
print('x:', x_.shape)
print('y:', y_.shape)

# 3: Shuffle
shuffle = np.hstack((y_, x_)).copy()
# print('before shuffle:', b4shuffle)
np.random.shuffle(shuffle)  # shuffle
# print('after shuffle:', b4shuffle.shape)
# print(b4shuffle)
# 4: Separate
sep = int(0.3 * len(x_))
train_x = shuffle[:sep, 1:].copy()
train_y = shuffle[:sep, 0:1].copy()
test_x = shuffle[sep:, 1:].copy()
test_y = shuffle[sep:, 0:1].copy()
print('train_x:', train_x.shape)
print('train_y:', train_y.shape)
print('test_x:', test_x.shape)
print('test_y:', test_y.shape)
# 5: CLF Model
dt = tree.DecisionTreeClassifier(max_depth=100)
dt.fit(train_x, train_y)
pred_y = dt.predict(test_x)
print('pred_y: ', pred_y)
print('test_y: ', test_y.T[0])
# 6: Accu
print("Accu：", metrics.precision_score(test_y.T[0], pred_y, average='micro'))
print("Recall：", metrics.recall_score(test_y.T[0], pred_y, average='macro'))
print("F1-score：", metrics.f1_score(test_y.T[0], pred_y, average='micro'))

end_time = time.time()
print('Total time:', end_time - start_time)
print(1)
