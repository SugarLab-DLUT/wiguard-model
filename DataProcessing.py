import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
import os

file_path = 'C:/Users/Guo Rui/Desktop/WiFi_CSI/npy' #数据当前目录 C:\Users\Guo Rui\Desktop\WiFi_CSI\npy
store_path = 'C:/Users/Guo Rui/Desktop/WiFi_CSI/npy2'#处理好的数据的存储目录

def filt(data):
    b, a = signal.butter(8, 0.5, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data, axis=1)  #data为要过滤的信号
    return filtedData

components_number = 20
pca = PCA(n_components=components_number)
fre = 60 #采样频率
for file_path, sub_dirs, filenames in os.walk(file_path):
    if filenames:  #there are files in root(file_path)
        filenames = sorted(np.array(filenames))
        for filename in filenames:
            print(filename)
            data = np.load(os.path.join(file_path, filename), allow_pickle=True)
            filtedData = filt(data)
            #50Hz，sample 长度5s，2s 有效数据
            for i in range(9000//(5*fre)): #250代表5s的数据
                data_5s = filtedData[:, 5*fre*i:5*fre*(i+1)]
                data_2s = data_5s[:, fre:3*fre]#(30, time) 变换到（time， 30）
                data_pca = data_2s
                #data_pca = pca.fit_transform(data_2s)
                data_pca = np.transpose(data_pca)
                newFileName = filename[:-4] + '_' + str(i+1)
                np.save(os.path.join(store_path, newFileName), data_pca)
                #print(data_pca.shape)
                #plt.plot(data_pca.reshape(-1))
                #plt.show()
