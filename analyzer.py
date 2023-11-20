import os
import csiread
import numpy as np
from numpy import ndarray
from scipy import signal
from sklearn.decomposition import PCA

csidata = csiread.Intel('dat/csi-temp.dat')
csidata.read()
scaled_csi: ndarray = csidata.get_scaled_csi()
print(scaled_csi)
print(scaled_csi.shape)

# (packet_num, subcarrier_num, rx_antenna_num, tx_antenna_num)
# lstm
# scaled_csi = np.abs(np.squeeze(scaled_csi[0, :, :]).T)
# see https://github.com/citysu/csiread/blob/master/examples/csishow.py for csi operation
# print(scaled_csi)
# # TODO ant
# ant1 = scaled_csi[:, 0]
# # ant2 = scaled_csi[:, 1]

# # data process
# components_number = 20
# pca = PCA(n_components=components_number)
# fre = 60  # 采样频率
# data = ant1

# b, a = signal.butter(8, 0.5, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
# filtedData = signal.filtfilt(b, a, data, axis=1)  # data为要过滤的信号
# # 50Hz，sample 长度5s，2s 有效数据
# for i in range(9000//(5*fre)):  # 250代表5s的数据
#     data_5s = filtedData[:, 5*fre*i:5*fre*(i+1)]
#     data_2s = data_5s[:, fre:3*fre]  # (30, time) 变换到（time， 30）
#     data_pca = data_2s
#     # data_pca = pca.fit_transform(data_2s)
#     data_pca = np.transpose(data_pca)
#     np.save(os.path.join('npy', 'csi-sample.npy'), data_pca)
#     # print(data_pca.shape)
#     # plt.plot(data_pca.reshape(-1))
#     # plt.show()
