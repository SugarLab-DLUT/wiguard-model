import os
import numpy as np
from scipy.io import loadmat

# 读取文件
files = os.listdir('C:/Users/Guo Rui/Desktop/WiFi_CSI/csi1.dat')
len_files = len(files)

for k in range(len_files):
    n = os.path.join('E:/CSI/data1/100/', files[k])
    csi_trace = loadmat(n)  # 你需要一个函数来读取你的数据文件，这里假设它是 .mat 文件

    first_ant_csi = []
    second_ant_csi = []
    # third_ant_csi = []

    for i in range(200):  # 这里是取的数据包的个数
        csi_entry = csi_trace[i]
        csi = get_scaled_csi(csi_entry)  # 提取csi矩阵，你需要一个函数来获取缩放后的 CSI
        csi = csi[0, :, :]
        csi1 = np.abs(np.squeeze(csi).T)  # 提取幅值(降维+转置)

        # 只取一根天线的数据
        first_ant_csi.append(csi1[:, 0])  # 直接取第一列数据(不需要for循环取)
        second_ant_csi.append(csi1[:, 1])
        # third_ant_csi.append(csi1[:, 2])

    l = files[k].name
    m1 = os.path.join('E:/CSI/data1/txt/12/', l, 'a.txt')
    m2 = os.path.join('E:/CSI/data1/txt/12/', l, 'b.txt')
    m3 = os.path.join('E:/CSI/data1/txt/12/', l, 'c.txt')

    np.savetxt(m1, first_ant_csi, delimiter=' ')
    np.savetxt(m2, second_ant_csi, delimiter=' ')
    # np.savetxt(m3, third_ant_csi, delimiter=' ')
