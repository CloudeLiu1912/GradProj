import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,roc_auc_score

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

rf = RandomForestClassifier()
rf.fit(train_x,train_y)
pre_test = rf.predict(test_x)


auc_score = roc_auc_score(test_y, pre_test)
pre_score = precision_score(test_y, pre_test)


print("auc_score,pre_score:", auc_score, pre_score)
