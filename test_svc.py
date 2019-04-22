import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics


# A(10, 10), xc=2, yc=2
Ax = 2 * np.random.randn(100) + 10
Ay = 2 * np.random.randn(100) + 10
A = np.vstack((Ax, Ay))

# B(2, 30), xc=4, yc=2
Bx = 4 * np.random.randn(100) + 2
By = 2 * np.random.randn(100) + 30
B = np.vstack((Bx, By))

# C(3, -3), xc=3, yc=3
Cx = 3 * np.random.randn(100) + 3
Cy = 3 * np.random.randn(100) - 3
C = np.vstack((Cx, Cy))

x_ = np.hstack((A, B, C)).reshape(300, 2)
y_ = np.array(([1] * 100, [2] * 100, [3] * 100)).reshape(1, 300)[0].T.reshape(300, 1)
print(x_.shape)
print(y_.shape)

shuffle = np.hstack((y_, x_)).copy()
np.random.shuffle(shuffle)  # shuffle
print('shuffle', shuffle)

sep = int(0.3 * len(x_))
train_x = shuffle[:sep, 1:].copy()
train_y = shuffle[:sep, 0:1].copy()
test_x = shuffle[sep:, 1:].copy()
test_y = shuffle[sep:, 0:1].copy()
print('train_x:', train_x.shape)
print('train_y:', train_y.shape)
print('test_x:', test_x.shape)
print('test_y:', test_y.shape)

svc0 = SVC(kernel='rbf', C=10, gamma=1)
# train_x0 = train_x.copy
# train_y0 = train_y.copy
# test_x0 = test_x.copy
# test_y0 = test_y.copy
svc0.fit(train_x, train_y)
pred_y = svc0.predict(test_x)
print('pred_y: ', pred_y)
print('test_y: ', test_y.T[0])

acc = pred_y-test_y.T[0]
num = 0
for i in acc:
    if i != 0:
        num += 1

print(acc)
print(1-num/300)

# print("Accu：", metrics.precision_score(test_y.T[0], pred_y, average='micro'))
# print("Recall：", metrics.recall_score(test_y.T[0], pred_y, average='macro'))
# print("F1-score：", metrics.f1_score(test_y.T[0], test_y, average='micro'))

plt.figure()

plt.scatter(Ax, Ay)
plt.scatter(Bx, By)
plt.scatter(Cx, Cy)
plt.show()

# print('Ax:', shuffle)
print('1')
