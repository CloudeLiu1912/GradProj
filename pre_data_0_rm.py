import numpy as np
import natsort
import time
import matplotlib.pyplot as plt
import pandas as pd
start_time = time.time()


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

np.set_printoptions(suppress=True)

ctrl_group = np.load('../data/ctrl_group.npy')
x = np.load('../data/spa_lines.npy')
y = np.loadtxt('../data/190305/1_001.spa', skiprows=1, usecols=1)
print(y)
#
# plt.plot(x, y-ctrl_group)
# plt.show()

all_ = np.load('../data/afterSTD.npy')
# print('all:', all_[0].shape)

r = np.vstack((x, y-ctrl_group, (y-0.1)/(ctrl_group-0.1), all_[0], y, ctrl_group))
f = []
for i in Ele:
    for j in i:
        if np.round(j, 2) in np.round(x, 2):
            f_n = np.argwhere(np.round(x, 2) == np.round(j, 2))
            f_ = r.T[f_n][0][0]
            print('j:', j, '   f_:', f_.shape)
            f.append(f_)
            # print(f.shape)
        else:
            print('error')
# print('r:', r.T)
print('f:', np.array(f))
np.savetxt('../data/0_1_1v.csv', f, delimiter=',')

# print('ctrl group:', ctrl_group.shape)
# print(ctrl_group)

end_time = time.time()
print('Total time: ', end_time-start_time)
print(1)
