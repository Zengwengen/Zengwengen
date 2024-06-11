import numpy as np
import matplotlib.pyplot as plt


def moving_average_filter(signal, window_size):
    window = np.ones(window_size)/window_size
    return np.convolve(signal, window, mode='same')


plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimSun']
t = np.linspace(0, 10, 100)
signal = np.sin(t) + np.random.randn(100)*0.2
filtered_signal = moving_average_filter(signal, window_size=30)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='原数据')
plt.plot(t, filtered_signal, label='滤波后数据')
plt.title('移动平均法滤波(M=30)')
plt.legend()
plt.grid(True)
plt.show()
