import os
import torch
import torch.nn.functional as torchF
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from Network import MyLSTM
from DataLoader import MyDataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

decay = 4e-5

def validating(model, validationset, kind="val"):
    with torch.no_grad():  # close grad tracking to reduce memory consumption

        total_correct = 0
        total_samples = 0
        model.eval()
        total_preds = []
        for batch in validationset:
            images, labels= batch
            #print(images.shape)
            preds = model(images)
            total_preds.append(preds.argmax(dim=1).numpy())
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

            total_samples += len(labels)

        val_acc = total_correct / total_samples
        print(kind + "_acc: ", total_correct / total_samples)
        # get confusion matrix elements [(labels, preds),...]#model.train()return val_acc
        total_preds = np.array(total_preds).reshape((-1,))
    return val_acc, total_preds

def main():
    print("testing...")
    file_path = 'C:/Users/Guo Rui/Desktop/WiFi_CSI/npy2' #存放某一类动作测试集的"文件夹"，不能是文件的路径
    Dataset = MyDataLoader(file_path, components_number=30)
    Dataset = torch.utils.data.DataLoader(Dataset, batch_size=32, shuffle=False) #文件夹中样本个数要大于 batch_size，测试一个样本时batch_size可设置为1
    model = torch.load('C:/Users/Guo Rui/Desktop/WiFi_CSI/models/model/model.pth') #存放模型的路径
    model.eval()
    acc, preds_labels = validating(model, Dataset, kind="test")
    print("Total test sample : ", len(preds_labels))
    print("检测结果 : ", )
    #print(preds_labels) #直接输出

    for idx, pred in enumerate(preds_labels): #加上“a"
        # print("sample {} is walking (a{})".format(idx, pred),end=" \n")
          print("用户跌倒(a0)。")
    #print(preds_labels)


if __name__ == '__main__':
    main()
