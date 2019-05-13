import numpy as np
import matplotlib.pyplot as plt


x = np.load('../data/0_pca_3600x54_95%.npy')
y = np.load('../data/ctrl_group.npy')

for i in x[1]:
    i = i.round(4)
    print(i)

# print(x)

# plt.rcParams['savefig.dpi'] = 300  #图片像素
# plt.rcParams['figure.dpi'] = 300  #分辨率

# plt.plot(x, y)
# plt.show()
