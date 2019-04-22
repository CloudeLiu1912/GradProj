import numpy as np
import os
import natsort
import time
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


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
Na = [589.592, 588.995]
Ni = [341.477, 352.454, 300.249, 232.003, 361.939, 305.082, 356.673, 347.254, 464.000, 547.691, 241.931]
P = [253.565, 213.618, 249.800, 214.914, 255.328]
Pb = [405.783, 368.348, 363.958, 283.306, 261.418, 280.199, 216.999, 241.174, 266.316, 239.379, 257.726]
Rb = []
S = [217.048]
Se = [203.985, 206.279]
Ti_ = [334.941, 368.520, 323.452, 376.132, 498.173, 365.459, 399.864, 430.396, 453.397, 468.192, 389.849]
V_ = [437.924, 411.178, 440.764, 370.400, 292.403, 439.386, 385.537, 290.882, 488.057, 426.864, 270.051, 454.539]
Zn = [213.856, 481.053, 472.216, 427.720, 202.548, 228.900, 334.502, 468.014, 206.200]
Ele = np.hstack((Cu, Fe, Mn, Zn, Al, Ca, Cd, Co, Cr, K, La, Li, Mg, Mo, Na, Ni, P, Pb, Rb, S, Se, Ti_, V_))
print('Ele.shape:', Ele.shape)
n = 0

files = os.listdir('data/190305')
file_list = natsort.natsorted(files)
spa = np.load('spa_lines.npy')
spa = np.round(spa, 3)
print('spa:', spa)
print('spa[0]', spa[0])
print('filelist: ', file_list)


def get_local_arg(p, spa_, eps=0.25):
    st = np.round(p-eps, 3)
    ed = np.round(p+eps, 3)
    print('st:', st, st in spa)
    print('ed:', ed, ed in spa)
    while st not in spa_:
        st = np.round(st-0.001, 3)
        # print('9-9', st)
    while ed not in spa_:
        ed = np.round(ed-0.001, 3)
    print('st:', st)
    print('ed:', ed)
    st_index = np.argwhere(spa_ == st)[0][0]
    ed_index = np.argwhere(spa_ == ed)[0][0]
    return st_index, ed_index


def get_section():
    section = []
    for ele in Ele:
        r = np.array(get_local_arg(ele, spa))
        # print('r:', spa[r[0]], spa[r[1]])
        section.append(r)
    section = np.array(section)
    print(section.shape)
    np.save('data/section_144x2_index.npy', section)


s = np.load('data/section_144x2_index.npy')

s_sum = []
for i in range(4):
    for j in range(1):
        m_max = []
        d_ = np.load('data/190305new/'+str(i+1)+'_'+str(j+1)+'.npy')
        plt.plot(spa, d_)
        for k in range(144):
            max_arg = np.argmax(d_[s[k, 0]:s[k, 1]])
            m_max.append(s[k, 0]+max_arg)
        m_max = np.reshape(m_max, [1, 144])
        # print('max_arg', m_max)
        print(str(i+1)+'_'+str(j+1)+'.npy')

        d = np.load('data/190305new/'+str(i+1)+'_'+str(j+1)+'.npy')
        for k in m_max[0]:
            s_sum.append(np.sum((d[k-10:k+10])))
        print('sum:', s_sum)
        plt.scatter(spa[m_max[0]], d[m_max[0]], c='r')
        plt.show()
# np.save('data/new_features_144x4000.npy', s_sum)

end_time = time.time()

print('Total time:', end_time-start_time)
print('Done!')
