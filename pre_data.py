import numpy as np
import matplotlib.pyplot as plt
import os
import natsort


c = [4.021, 3.608, 3.986, 1.491, 3.618, 3.046, 3.535, 3.178,
     8.664, 8.481, 8.482, 7.598, 8.971, 8.973, 8.226, 8.386, 9.303, 9.306,
     6.356, 6.132, 7.806, 8.324, 8.267, 7.941, 8.927, 7.486,
     5.707, 4.946, 9.612, 8.944, 6.528, 6.772, 7.593, 7.238, 7.621, 8.494,
     6.656, 4.576, 5.585, 4.826]

Cu = [324.754, 327.396, 249.218, 223.008, 510.554]
Fe = [496.610, 343.987, 371.841, 302.049, 476.30, 259.940, 381.765, 248.265, 278.619, 404.461, 389.971]
Mn = [279.482, 257.610, 259.373, 260.567, 403.076]

Al = [396.068, 338.289]
Ca = [393.366, 422.673, 396.847]
Cd = [228.802, 326.106, 226.502, 508.582, 214.438, 479.992, 361.051]
Co = [241.289, 345.350, 251.102, 351.835]
Cr = [359.431, 425.435, 279.582, 285.322, 520.844, 570.6, 250.733, 275.939]
K = [404.721]
La = [324.513, 333.749, 379.478, 408.672, 398.852, 424.999, 428.679, 492.098, 229.736]
Li = [460.286, 497.170, 223.600, 447.200, 323.263, 413.256, 274.119, 395.900, 274.118]
Mg = [279.553, 285.213, 280.270, 383.826, 202.582]
Mo = [379.825, 386.411, 319.397, 390.296, 277.540, 268.414, 263.876, 476.019, 414.355, 473.144, 253.846]
Na = [589.592, 588.995, 196.500]
Ni = [341.477, 352.454, 300.249, 232.003, 361.939, 305.082, 356.673, 347.254, 464.000, 547.691, 241.931]
P = [253.565, 213.618, 249.800, 214.914, 255.328]
Pb = [405.783, 368.348, 363.958, 283.306, 261.418, 280.199, 216.999, 241.174, 266.316, 239.379, 257.726]
Rb = []
S = [217.048]
Se = [196.090, 203.985, 206.279]
Ti_ = [334.941, 368.520, 323.452, 376.132, 498.173, 365.459, 399.864, 430.396, 453.397, 468.192, 389.849]
V_ = [437.924, 411.178, 440.764, 370.400, 292.403, 439.386, 385.537, 290.882, 488.057, 426.864, 270.051, 454.539]
Zn = [213.856, 481.053, 472.216, 427.720, 202.548, 228.900, 334.502, 468.014, 206.200]


Ele = [Cu, Fe, Mn, Zn, Al, Ca, Cd, Co, Cr, K, La, Li, Mg, Mo, Na, Ni, P, Pb, Rb, S, Se, Ti_, V_]


def get_aver_control_group(directory='data/190305', key=True):  # key:1-需要读取，0-不需要重新读取
    if not key:
        spa_lines = np.load('spa_lines.npy')
        aver = np.load('ctrl_group.npy')
    else:
        files = os.listdir(directory)
        file_list = natsort.natsorted(files)
        print('filelist: ', file_list)
        sum = []
        num = 0
        for file in file_list:
            if not int(file[0]):
                num += 1
                d_tmp = np.loadtxt(directory + '/' + file, skiprows=1)
                if int(file[3:5]) == 1:
                    sum = d_tmp
                    print(2)
                else:
                    sum += d_tmp
                    print(3)
            # print(1)
        spa_lines = np.delete(sum/num, obj=1, axis=1).flatten()  # axis=1, delete the column 1
        aver = np.delete(sum/num, obj=0, axis=1).flatten()  # axis=1, delete the column 0
        np.save('spa_lines.npy', spa_lines)
        np.save('ctrl_group.npy', aver)
    return spa_lines, aver


def get_label(x):
    if x <= 8:
        return 1
    elif x <= 18:
        return 2
    elif x <= 26:
        return 3
    elif x <= 36:
        return 4
    else:
        return 5


def get_wavelet_cal(its, spa, f_line, eps=1.0):
    min_ = f_line-eps
    max_ = f_line+eps
    sum_ = 0
    for a in spa:
        if (a >= min_) and (a <= max_):
            index_ = np.argwhere(spa == a)[0][0]
            sum_ += its[index_]  # [index][1]:the spectral intensity
    return sum_


def get_valid_data(directory='data/190305', ctrl_group=0, key=True):  # key:1-需要读取，0-不需要重新读取
    if not key:
        data_ = np.load('data.npy')
        label_ = np.load('label.npy')
    else:
        data_ = []
        label_ = []

        files = os.listdir(directory)
        file_list = natsort.natsorted(files)
        print('filelist: ', file_list)
        index = 0

        for file in file_list:
            if int(file[0]):
                index += 1
                d_tmp = np.loadtxt(directory+'/'+file, skiprows=1)
                d_tmp = np.delete(d_tmp, obj=0, axis=1).flatten()

                # 1 : minus control group
                d_tmp = d_tmp - ctrl_group

                # 2 : zero-centered
                d_tmp = d_tmp/c[int(index/100)] * 10
                print(file, 'data:', d_tmp)
                np.save(file+'.npy', d_tmp)
                data_.append(d_tmp)

                l_tmp = get_label(index/100)
                print(file, 'label:', l_tmp)
                label_.append(l_tmp)

        np.save('label.npy', label_)
        print(1)

    return data_, label_


def get_features(directory='data/190305', key=False):
    if not key:  # False
        ret = np.load('new_data.npy')
        return ret
    else:  # True
        files = os.listdir(directory)
        file_list = natsort.natsorted(files)
        print('file_list: ', file_list)
        features_1 = []
        for file in file_list:
            its_ = np.loadtxt(directory + '/' + file, skiprows=1, usecols=1)
            spa_ = np.loadtxt(directory + '/' + file, skiprows=1, usecols=0)
            print('file:', file)
            for i in Ele:
                for j in i:
                    f_tmp = get_wavelet_cal(its_, spa_, j, eps=1.0)
                    features_1.append(f_tmp)
            np.save(file+'.npy', features_1)
            pass
        np.save('new_data.npy', features_1)
    return features_1


def get_peak_features():
    return 0


def main():
    # spa_lines, ctrl_g = get_aver_control_group(key=False)
    # print(spa_lines)
    # print(ctrl_g)
    # data_v, label_v = get_valid_data(ctrl_group=ctrl_g, key=False)
    get_features(key=True)

    print(1)


if __name__ == '__main__':
    main()
