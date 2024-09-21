import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import csiread

CLIP_SIZE = 380  # 截取的数据长度


class CSIDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        air_files = self.read_folder(os.path.join(data_path, 'air'))
        fall1_files = self.read_folder(
            (os.path.join(data_path, 'fall', 'fall1')))
        fall2_files = self.read_folder(
            (os.path.join(data_path, 'fall', 'fall2')))
        walk_files = self.read_folder(os.path.join(data_path, 'walk'))
        fall_files = fall1_files + fall2_files

        air_data = self.read_files(air_files)
        fall_data = self.read_files(fall_files)
        walk_data = self.read_files(walk_files)

        air_data = self.cut_data_len(air_data)
        fall_data = self.cut_data_len(fall_data)
        walk_data = self.cut_data_len(walk_data)

        air_data = self.sum_data(air_data)
        fall_data = self.sum_data(fall_data)
        walk_data = self.sum_data(walk_data)

        csis = air_data + fall_data + walk_data
        self.labels = [0] * len(air_data) + [1] * \
            len(fall_data) + [2] * len(walk_data)
        self.amplitudes = self.compute_amplitude(csis)

        self.amplitudes = [(amplitude - np.mean(amplitude)) /
                           np.std(amplitude) for amplitude in self.amplitudes]

    def read_folder(self, dir_path):
        """
        返回文件夹中所有.dat文件的路径
        :param dir_path: 文件夹路径
        :return: dat_file_paths 一个列表,列表中的每个元素是一个数据文件的路径
        """
        files = os.listdir(dir_path)
        dat_file_paths = []
        for file in files:
            if file.endswith(".dat"):
                dat_file_paths.append(os.path.join(dir_path, file))
        return dat_file_paths

    def read_files(self, data_paths):
        """
        读取所有.dat文件中的CSI数据
        :param data_paths: 一个列表,列表中的每个元素是一个数据文件的路径
        :return: csi_data 一个列表,列表中的每个元素是一个数据文件的CSI数据
        """
        csi_data = []
        for data_path in data_paths:
            csi = csiread.Intel(data_path, if_report=False)
            csi.read()
            shape = csi.csi.shape
            csi_np = np.array(csi.csi).reshape(
                shape[0], shape[1], shape[2], shape[3])
            csi_np = np.transpose(csi_np, (0, 3, 2, 1))
            csi_data.append(csi_np)
        return csi_data

    def cut_data_len(self, data_list, clip_size=CLIP_SIZE):
        """
        截取每一个数据的长度为clip_size，从数据的中间截取，如果数据长度小于clip_size，则丢弃
        :param data_list: 一个列表,列表中的每个元素是一个数据文件的CSI数据
        :param clip_size: 截取的长度
        """
        new_data_list = []
        for data in data_list:
            if len(data) >= clip_size:
                start = len(data) // 2 - clip_size // 2
                new_data_list.append(data[start: start + clip_size])
        return new_data_list

    def sum_data(self, data_list):
        """
        将数据列表中的数据合并
        :param data_list: 一个列表,列表中的每个元素是一个数据文件的CSI数据，shape: (clip_size, tx, rx, subcarries)
        :return: 合并后的数据, shape: (clip_size, subcarries)，原先的数据可以分出 tx * rx 个子数据，将子数据数值相加

        data_list = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        print(np.sum(data_list, axis=(1, 2)))  # [21 57]
        """
        new_data_list = []
        for data in data_list:
            new_data_list.append(np.sum(data, axis=(1, 2)))
        return new_data_list

    def compute_amplitude(self, csis_data):
        """
        使用NumPy的向量化操作计算CSI数据的幅度
        :param csis_data: 一个列表,列表中的每个元素是一个数据文件的CSI数据, shape: (clip_size, subcarries)
        :return: amplitude_data 一个列表，列表中的每个元素是对应的振幅数据, shape: (clip_size, subcarries)
        """
        amplitude_data = [np.abs(csi) for csi in csis_data]  # 计算CSI数据的幅度，即取绝对值
        return amplitude_data

    def __len__(self):
        return len(self.amplitudes)

    def __getitem__(self, idx):
        return self.amplitudes[idx], self.labels[idx]


def process_single_dat(dat_path):
    """
    处理单个.dat文件，截取数据的中间部分，合并不同发射天线和接收天线的数据，计算振幅数据，并进行标准化
    :param dat_path: .dat文件路径
    :return: amplitude_data 振幅数据，shape: (clip_size, subcarries)
    """
    csi = csiread.Intel(dat_path, if_report=False)
    csi.read()
    shape = csi.csi.shape
    csi_np = np.array(csi.csi).reshape(shape[0], shape[1], shape[2], shape[3])
    csi_np = np.transpose(csi_np, (0, 3, 2, 1))
    # 截取每一个数据的长度为clip_size，从数据的中间截取，如果数据长度小于clip_size，则报错退出
    if len(csi_np) < CLIP_SIZE:
        raise ValueError('The length of data is less than CLIP_SIZE')
    start = len(csi_np) // 2 - CLIP_SIZE // 2
    csi_np = csi_np[start: start + CLIP_SIZE]
    csi_np = np.sum(csi_np, axis=(1, 2))
    amplitude_data = np.abs(csi_np)
    amplitude_data = (amplitude_data - np.mean(amplitude_data)
                      ) / np.std(amplitude_data)
    return amplitude_data


if __name__ == '__main__':
    BATCH_SIZE = 2
    csi_dataset = CSIDataset('./data')
    total_size = len(csi_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        csi_dataset, [train_size, val_size])  # 分割数据集
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 统计训练集和验证集的每种类别的数量
    class_num = [0, 0, 0]
    for data, label in train_loader:
        for l in label:
            class_num[l] += 1

    print('Train dataset:')
    print('Air:', class_num[0])
    print('Fall:', class_num[1])
    print('Walk:', class_num[2])

    class_num = [0, 0, 0]
    for data, label in val_loader:
        for l in label:
            class_num[l] += 1

    print('Validation dataset:')
    print('Air:', class_num[0])
    print('Fall:', class_num[1])
    print('Walk:', class_num[2])
