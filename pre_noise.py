import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

# 读取txt数据
a = np.loadtxt('data/190305/0_001.spa', delimiter=' ', skiprows=1)
x = a[:, 0]
y = a[:, 1]

# 对强度信号进行傅里叶变换
s = fft(y)

# 为了滤除高频噪声信号，采用截断函数可以做到这一点
# 根据傅里叶变换的结果设置合适的截断函数
m = len(s)
n = 50
cutfun = np.ones([m, 1])
cutfun[20:m - 20] = 0
ss = s
ss[n:m - n] = 0  # 对傅里叶变换信号做截断
f = ifft(ss)  # 逆傅里叶变换
real_f = np.real(f)  # 取实部

plt.plot(x, y, 'y')
plt.plot(x, real_f, 'r', lw=3)

plt.show()
