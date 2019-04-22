import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import natsort
from sklearn.svm import SVC
from sklearn import metrics


def main():
    # 1: Row data
    all_data = np.load('data/pca_4000x6.npy')
    print('all.shape', all_data.shape)
    re_all_data = all_data.reshape([4000, 6])
    row_label = np.load('label.npy')

    # 2: Slide
    blank_ = re_all_data[:100].copy()
    x_ = re_all_data[100:3700].copy()  # ginseng
    soil_ = re_all_data[3700:].copy()  # soil
    y_ = row_label[:3600].reshape([3600, 1]).copy()
    print('blank:', blank_.shape)
    print('x:', x_.shape)
    print('y:', y_.shape)

    # 3: Shuffle
    shuffle = np.hstack((y_, x_)).copy()
    # print('before shuffle:', b4shuffle)
    np.random.shuffle(shuffle)  # shuffle
    # print('after shuffle:', b4shuffle.shape)
    # print(b4shuffle)

    # 4: Separate
    sep = int(0.3*len(x_))
    train_x = shuffle[:sep, 1:].copy()
    train_y = shuffle[:sep, 0:1].copy()
    test_x = shuffle[sep:, 1:].copy()
    test_y = shuffle[sep:, 0:1].copy()
    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)
    print('test_x:', test_x.shape)
    print('test_y:', test_y.shape)

    train_x3 = train_x[train_y[:, 0] >= 3].copy()
    test_x3 = test_x[test_y[:, 0] >= 3].copy()
    # print('trainx2: ', train_x2.shape)
    train_y3 = train_y[train_y[:] >= 3].copy()
    test_y3 = test_y[test_y[:] >= 3].copy()
    # print('trainy2: ', train_y2.shape)

    # 5 // SVC - poly
    svc0 = SVC(kernel='poly', C=0.5, gamma=100)
    # train_x0 = train_x.copy
    # train_y0 = train_y.copy
    # test_x0 = test_x.copy
    # test_y0 = test_y.copy
    svc0.fit(train_x, train_y)
    pred_y = svc0.predict(test_x)
    print('pred_y: ', pred_y)
    print('test_y: ', test_y.T[0])

    print("Accu：", metrics.precision_score(test_y.T[0], pred_y, average='micro'))
    print("Recall：", metrics.recall_score(test_y.T[0], pred_y, average='macro'))
    print("F1-score：", metrics.f1_score(test_y.T[0], pred_y, average='micro'))

    # 6 // SVC - rbf
    svc1 = SVC(kernel='rbf', C=10, gamma=0.1)
    # train_x0 = train_x.copy
    # train_y0 = train_y.copy
    # test_x0 = test_x.copy
    # test_y0 = test_y.copy
    svc1.fit(train_x, train_y)
    pred_y1 = svc1.predict(test_x)
    print('pred_y: ', pred_y1)
    print('test_y: ', test_y.T[0])

    print("Accu：", metrics.precision_score(test_y.T[0], pred_y1, average='micro'))
    print("Recall：", metrics.recall_score(test_y.T[0], pred_y1, average='macro'))
    print("F1-score：", metrics.f1_score(test_y.T[0], pred_y1, average='micro'))


    # # 5: Build SVM1 : Separate: 1 vs 234
    # svm1 = SVC(kernel='rbf', C=1, gamma=0.5)
    # train_y1 = train_y.copy()  # 1234
    # test_y1 = test_y.copy()
    # train_y1[train_y1[:] >= 2] = 0  # IN y，234 as 0，1 as 1
    # test_y1[test_y1[:] >= 2] = 0
    # svm1.fit(train_x, train_y1)  # fit(&1, &234)
    # pred1 = svm1.predict(test_x)
    # print('pred1: ', pred1)

    # # 6: Build svm2 : Separate: 2 vs 34
    # svm2 = SVC(kernel='rbf', C=1, gamma=0.5)
    # train_x2 = train_x[train_y[:, 0] >= 2].copy()  # 234
    # train_y2 = train_y[train_y[:] >= 2].copy()
    # test_x2 = test_x[test_y[:, 0] >= 2].copy()
    # test_y2 = test_y[test_y[:] >= 2].copy()
    # train_y2[train_y2[:] >= 3] = 0  # 2 vs 34
    # test_y2[test_y2[:] >= 3] = 0
    # svm2.fit(train_x2, train_y2)
    # pred2 = svm2.predict(test_x2)

    # print('pred1: ', pred1)
    # error = pred1 - np.transpose(test_y1)
    # print('error1: ', error[:10])
    # for i in error.flat:
    #     if i == 0.:
    #         count = count + 1
    # print('acc_n: ', count)
    # #
    #
    # # 6: Predict & calculate acc
    # #
    # #
    # # print('acc: ', count / len(pred))

    print('Done!')
    print(1)


if __name__ == '__main__':
    main()
