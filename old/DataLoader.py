import numpy as np
import os
import torch
from torch.utils.data import Dataset
#np.random.seed(1)


class MyDataLoader(Dataset):
    def __init__(self, file_path=None, CWdata=True, add_noise=False, components_number=10): #
        self.file_path = file_path
        self.CWdata = CWdata
        self.add_noise = add_noise
        self.components_number = components_number

        self.images, self.labels,  = self.load_file2MFCC(self.file_path,)

    def addGuassiannoise(self, feature, mean=0, var=0.001):
        shape = feature.shape
        noise = np.random.normal(loc=mean, scale=var, size=shape)
        feature += noise
        feature = 2 * (feature - np.min(feature)) / (np.max(feature) - np.min(feature)) - 1
        return feature

    def load_file2MFCC(self, file_path):
        wave_feature = []
        labels = []


        #labsIndex = []
        for file_path, sub_dirs, filenames in os.walk(file_path):
            if filenames:  #there are files in root(file_path)
                filenames = sorted(np.array(filenames))
                for filename in filenames:
                    ###file_list.append(os.path.join(file_path, filename))
                    #wave_names.append(file_path[path_len:] + '_' + filename)
                    #print(filename)
                    feature = np.load(os.path.join(file_path, filename), allow_pickle=True)
                    #if self.CWdata:
                        #feature = feature.reshape(self.components_number, 30) #数据是30（30个载波） * n_components=10

                    #feature = feature[0,:,:]
                    feature = 2*(feature-np.min(feature)) / (np.max(feature)-np.min(feature)) - 1 #scale to [-1,1]
                    if self.add_noise:
                        feature = self.addGuassiannoise(feature)
                    wave_feature.append(feature)
                    print("the number of file:", len(wave_feature))
                    labels.append(int(filename[4])) #公开数据集=5， 自采数据集=4

                    

        #shuffle the numpy, then convert into tensor
        wave_feature = np.array(wave_feature)
        labels = np.array(labels)

        ind_test = np.arange(len(labels))
        np.random.shuffle(ind_test)
        wave_feature, labels = wave_feature[ind_test], labels[ind_test]
        
        wave_feature = torch.Tensor(wave_feature)
        labels = torch.Tensor(labels).long()  # 训练时需要labels为long型

        return wave_feature, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

if __name__ == '__main__':
    file_path = 'F:\WiAR-master\data1\ProcessedData'
    dataset = MyDataLoader(file_path)
    #len(dataset.images)
    print(dataset.images[0])
