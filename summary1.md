## 将数据处理代码从 Matlab 转移到 Python

首先分析原先的 csi 数据处理代码：

```matlab
files = dir('C:\Users\Guo Rui\Desktop\WiFi_CSI\csi1.dat');
len=length(files);
for k=1:len
  n=strcat('E:\CSI\data1\100\',files(k).name);
  csi_trace = read_bf_file(n);
 for i=1:200%这里是取的数据包的个数
        csi_entry = csi_trace{i};
        csi = get_scaled_csi(csi_entry); %提取csi矩阵    
        csi =csi(1,:,:);
        csi1=abs(squeeze(csi).');          %提取幅值(降维+转置)

        %只取一根天线的数据
        first_ant_csi(:,i)=csi1(:,1);           %直接取第一列数据(不需要for循环取)
        second_ant_csi(:,i)=csi1(:,2);
       % third_ant_csi(:,i)=csi1(:,3);
 end
   l=files(k).name;
   m1=strcat('E:\CSI\data1\txt\12\',l,'a.txt');
   m2=strcat('E:\CSI\data1\txt\12\',l,'b.txt');
   m3=strcat('E:\CSI\data1\txt\12\',l,'c.txt');
   dlmwrite(m1,first_ant_csi,'delimiter',' ')
   dlmwrite(m2,second_ant_csi,'delimiter',' ')
   dlmwrite(m3,third_ant_csi,'delimiter',' ')
end

%画第一根天线的载波
%plot(first_ant_csi.')
%plot(second_ant_csi.')
%plot(third_ant_csi.')
```

这段代码主要是从文件中读取 .dat 文件并解析为矩阵，然后提取了其中的幅值，并转换成 txt 文本以便后面 `data.py` 和 `DataProcessing.py` 处理。这里不仅没有完全使用所有的天线数据，还增加了流程复杂性，考虑将原本的功能合并到一个 Python 文件中。

在 PyPI 上找到了一个比较好用的 CSI 解析库 csiread，其封装了使用 Matlab 处理数据的操作（并且比 Matlab 的处理速度更快），还提供了基本的降维转置功能。此外，还提供了实时绘图功能方便调试测试。

> A **fast** channel state information parser for Intel, Atheros, Nexmon, ESP32 and PicoScenes in Python.
>
> - Full support for [Linux 802.11n CSI Tool](https://dhalperi.github.io/linux-80211n-csitool/), [Atheros CSI Tool](https://wands.sg/research/wifi/AtherosCSI/), [nexmon_csi](https://github.com/seemoo-lab/nexmon_csi) and [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)
> - Support for [PicoScenes](https://ps.zpj.io/) is **experimental**.
> - At least 15 times faster than the implementation in Matlab
> - Real-time parsing and visualization.
>
> **real-time plotting**
>
> [![real-time plotting](https://github.com/citysu/csiread/raw/master/docs/sample2.png)](https://github.com/citysu/csiread/blob/master/docs/sample2.png)

同时，决定不再使用 `data.py` 和 `DataProcessing.py` 的代码。原本的处理代码是从文件夹中读取多个 csi 文件合并进行计算。实际上可以配合新的 `recv.py` 接收代码实现实时检测等功能。故决定利用 csiread 库重写数据处理部分，正在等待与模型进行数据对接。

```python
import os
import csiread
import numpy as np
from numpy import ndarray
from scipy import signal
from sklearn.decomposition import PCA

csidata = csiread.Intel('dat/csi.dat')
csidata.read()
scaled_csi: ndarray = csidata.get_scaled_csi()
print(scaled_csi)
print(scaled_csi.shape)
# (packet_num, subcarrier_num, rx_antenna_num, tx_antenna_num)
# see https://github.com/citysu/csiread/blob/master/examples/csishow.py for csi operation

# scaled_csi = np.abs(np.squeeze(scaled_csi[0, :, :]).T)
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

```

使用 `csidata.read()` 和 `csidata.get_scaled_csi()` 后可以直接获得 numpy 矩阵格式的csi数据，其四个维度分别为 `(数据包数量, 子载波数量, 接收天线数量, 发射天线数量)`。代码后方注释的部分是原先的滤波代码，之后可以使用 csiread 内置函数替代。

在 `csiread.Intel('dat/csi.dat')` 中可以追加参数设置发射和接收天线数量，默认为 2 发 3 收。

样例 `scaled_csi` 的数据

```python
# connector_log=0x1
# 317 0xbb packets parsed
# (317, 30, 3, 2)
# 317 包, 30 子载波, 3 接收, 2 发射
[
 # 每个数据包
 [
  # 每个子载波
  #  发射天线 1 --------------  发射天线 2 --------------
  #  幅度 ------- 相位 -------  幅度 ------- 相位 -------
  [[ -8.48525955+17.67762407j  -9.19236452 -3.53552481j]  # 接收天线 1
   [ -7.77815459 +4.24262978j   9.19236452+10.60657444j]  # 接收天线 2
   [  0.         +0.j           0.         +0.j        ]] # 接收天线 3

  [[ 20.50604392+14.14209925j -11.3136794 +13.43499429j]
   [  2.12131489+12.02078437j  17.67762407 -9.89946948j]
   [  0.         +0.j           0.         +0.j        ]]

  [[ 21.21314888-18.38472903j  12.02078437+12.72788933j]
   [ 13.43499429 +0.70710496j  -4.94973474-21.92025384j]
   [  0.         +0.j           0.         +0.j        ]]
  ... # 剩余 27 个子载波
 ]
 ... # 剩余 316 个数据包
]
```

其中，每一个信道响应都是一个复数，实部代表信号的幅度，虚部代表信号的相位。