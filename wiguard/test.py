import sys
import torch
import logging
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from wiguard.dataset import process_single_csv_file, process_single_csv_data, CSIDataset
from wiguard.model.Transformer import Transformer
# from wiguard import config

torch.manual_seed(0)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Device: {}".format(device))

CLIP_SIZE = 100  # 截取的数据长度
SUBCARRIES = 64  # 子载波数
LABELS_NUM = 3
EPOCHS_NUM = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 2
ENC_SEQ_LEN = 6  # 编码器序列长度
DEC_SQL_LEN = 4  # 解码器序列长度
DIM_VAL = 32  # value的维度
DIM_ATTN = 16  # attention的维度
N_HEADS = 4  # 多头注意力的头数
N_ENCODER_LAYERS = 4  # 编码器层数
N_DECODER_LAYERS = 4  # 解码器层数
WEIGHT_DECAY = 0  # 权重衰减

pth_path = "./models/model0.pth"

model = Transformer(dim_val=DIM_VAL,
                    dim_attn=DIM_ATTN,
                    input_size=SUBCARRIES,
                    dec_seq_len=DEC_SQL_LEN,
                    out_seq_len=LABELS_NUM,
                    n_decoder_layers=N_DECODER_LAYERS,
                    n_encoder_layers=N_ENCODER_LAYERS,
                    n_heads=N_HEADS)
model.float().to(device)
if (not torch.cuda.is_available()):
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
else:
    model.load_state_dict(torch.load(pth_path))


def test_predict_file(csv_path):
    '''
    使用存放于csv文件里的csi数据，批量处理，用于模型训练
    '''
    amplitude_data = process_single_csv_file(csv_path)

    amplitude_data = torch.tensor(amplitude_data).float().to(device)
    amplitude_data = amplitude_data.unsqueeze(0)

    output = model(amplitude_data)
    pred = F.log_softmax(output, dim=1).argmax(dim=1)
    # print(pred)
    if pred[0] == 0:
        res = 'empty'
    elif pred[0] == 1:
        res = 'fall'
    else:
        res = 'walk'
    print(res)
    return res

def test_predict(csi_data):
    '''
    使用一定数量的以字符串形式传入的csi数据，进行预测。
    '''
    amplitude_data = process_single_csv_data(csi_data)
    amplitude_data = torch.tensor(amplitude_data).float().to(device)
    amplitude_data = amplitude_data.unsqueeze(0)

    output = model(amplitude_data)
    pred = F.log_softmax(output, dim=1).argmax(dim=1)
    # print(pred)
    if pred[0] == 0:
        res = 'empty'
    elif pred[0] == 1:
        res = 'fall'
    else:
        res = 'walk'
    print(res)
    return res

def test_train():


    # 准备数据集
    csi_dataset = CSIDataset('../data')
    # print(len(csi_dataset))
    total_size = len(csi_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    # print(train_size, val_size)
    train_dataset, val_dataset = random_split(
        csi_dataset, [train_size, val_size])  # 分割数据集

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # print(len(train_loader), len(val_loader))
    logging.info("Train size: {}, val size: {}".format(len(train_loader), len(val_loader)))

    # Loss Function CrossEntropy
    loss_fn = CrossEntropyLoss()

    # Optimizer Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # outputs: (batch_size, num_label)

    writer = SummaryWriter(log_dir='./logs_train')

    total_train_step = 0
    total_val_step = 0
    for epoch in range(EPOCHS_NUM):
        model.train()
        total_train_loss = 0
        logging.info("Epoch: {}".format(epoch))
        for data, label in train_loader:
            data.float().to(device)
            label.float().to(device)

            # print(data.shape, label.shape)
            # print(model(data.to(device)))

            outputs = model(data)
            # print(pred)
            loss = loss_fn(outputs, label).float()
            total_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_train_step += 1


        model.eval()
        total_valid_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data, label in val_loader:
                data.float().to(device)
                label.float().to(device)
                outputs = model(data)
                total_accuracy += outputs.argmax(dim=1).eq(label).sum()
                valid_loss = loss_fn(outputs, label).float()
                total_valid_loss += valid_loss



        print("step: {}".format(total_train_step), "train loss: {}".format(total_train_loss/len(train_loader)))
        print("step: {}".format(total_train_step), "valid loss: {}".format(total_valid_loss/len(val_loader)))
        print("step: {}".format(total_train_step), "accuracy: {}".format(total_accuracy/val_size))

        writer.add_scalar('Loss/train', total_train_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/val', total_valid_loss/len(val_loader), epoch)
        writer.add_scalar('Accuracy/val', total_accuracy/val_size, epoch)

    torch.save(model.state_dict(), '../models/model0.pth')



if __name__ == '__main__':

    '''
    对单个csv文件预测
    '''
    #test_predict_file(sys.argv[1])

    '''
    利用以字符串形式传入的数据
    '''
    # data = ['[0,0,0,0,0,0,0,0,0,0,0,0,-2,-14,-2,-14,-2,-14,-2,-12,-2,-11,-1,-9,0,-7,1,-5,2,-3,4,-1,5,1,6,2,8,3,9,4,10,4,11,4,11,3,11,3,11,3,10,2,8,2,6,1,4,1,2,1,0,2,-3,2,0,0,-7,4,-8,6,-9,6,-10,7,-11,8,-11,8,-10,8,-9,8,-8,7,-6,6,-4,4,-4,2,-2,0,-1,-2,0,-4,1,-6,1,-8,1,-10,1,-12,1,-13,1,-13,1,-13,1,-12,0,-11,0,-9,-1,-7,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,-10,-10,-10,-11,-10,-10,-9,-9,-8,-8,-7,-7,-5,-6,-3,-5,0,-4,2,-3,4,-3,6,-3,8,-3,9,-3,10,-4,11,-4,11,-5,10,-5,10,-5,9,-5,7,-4,6,-3,4,-2,2,-1,1,1,0,3,0,0,-2,7,-3,9,-3,10,-3,12,-3,13,-3,12,-2,11,-2,11,-1,9,-1,8,-1,6,-1,3,-1,1,-2,-2,-2,-3,-3,-6,-4,-8,-5,-9,-6,-10,-7,-11,-7,-11,-7,-10,-7,-9,-6,-8,-6,-7,-5,-5,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,3,2,1,3,-2,5,-3,7,-5,9,-5,11,-5,12,-5,13,-5,14,-5,14,-4,13,-3,12,-3,10,-3,8,-3,6,-3,3,-3,1,-4,-2,-5,-4,-7,-6,-8,-7,-10,-8,-11,-8,-12,-8,-12,-9,-13,-7,0,0,-11,-6,-9,-5,-7,-4,-5,-3,-2,-3,0,-3,3,-3,6,-3,8,-3,9,-4,10,-5,11,-6,12,-6,12,-6,11,-7,10,-7,8,-6,6,-6,4,-4,2,-3,0,-1,-1,1,-3,3,-4,5,-5,6,-5,7,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,-5,-13,-5,-13,-5,-12,-5,-11,-4,-10,-4,-9,-2,-7,-1,-5,1,-3,3,-2,4,-1,6,0,8,1,8,1,10,1,10,1,11,1,10,1,10,1,9,0,7,0,5,0,4,0,2,1,0,2,-2,3,0,0,-5,5,-7,6,-8,7,-8,8,-8,8,-8,8,-7,8,-6,7,-5,6,-4,5,-3,3,-2,1,-1,0,-1,-3,0,-4,0,-6,1,-8,0,-10,0,-11,-1,-11,-1,-12,-1,-11,-1,-10,-1,-9,-2,-8,-2,-6,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,4,-1,3,1,2,5,1,7,1,10,2,12,2,13,3,14,3,14,3,14,4,13,3,12,3,10,2,8,1,6,-1,4,-2,2,-4,0,-6,-1,-8,-2,-10,-3,-12,-3,-14,-3,-14,-3,-14,-3,-14,-2,0,0,-12,-1,-10,-1,-8,-1,-6,-1,-2,-2,0,-3,3,-4,5,-5,6,-6,8,-7,9,-8,9,-9,9,-9,9,-9,9,-10,7,-9,6,-8,4,-7,3,-5,1,-4,0,-1,-2,1,-3,3,-3,5,-4,7,-4,8,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,0,-4,2,-3,5,-3,7,-2,9,-2,9,-2,11,-2,12,-2,12,-2,12,-2,11,-2,9,-2,8,-2,7,-1,4,0,3,1,0,2,-1,4,-2,5,-3,6,-4,8,-4,9,-4,10,-4,11,-4,11,-4,10,0,0,-3,9,-2,7,-2,5,-2,3,-2,1,-2,-2,-2,-4,-3,-6,-4,-8,-4,-9,-5,-10,-6,-10,-6,-11,-6,-10,-6,-9,-6,-8,-6,-7,-5,-5,-4,-4,-3,-2,-1,0,0,1,2,3,3,4,5,4,5,5,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,-13,-7,-13,-7,-12,-7,-11,-6,-10,-5,-9,-5,-6,-5,-4,-4,-1,-4,1,-4,3,-5,6,-5,7,-6,8,-6,9,-6,10,-7,10,-8,9,-7,8,-7,7,-6,5,-5,4,-4,3,-3,2,0,1,2,0,4,0,0,-1,8,-1,9,-1,11,-1,12,0,12,0,12,0,11,1,10,1,9,1,7,0,5,0,2,-1,0,-2,-2,-2,-4,-4,-6,-4,-8,-6,-9,-7,-10,-8,-10,-9,-10,-8,-9,-8,-8,-7,-7,-7,-6,-6,-4,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,-3,13,-3,13,-3,13,-3,11,-3,9,-3,7,-3,5,-3,2,-4,0,-4,-3,-5,-5,-5,-7,-6,-8,-7,-9,-7,-10,-8,-10,-8,-10,-8,-9,-7,-9,-7,-8,-6,-6,-4,-5,-3,-4,-1,-3,2,-1,3,0,0,0,7,1,9,1,10,1,12,1,12,2,12,1,11,1,10,0,8,0,7,-1,5,-1,2,-1,0,0,-2,0,-4,0,-6,1,-8,1,-10,2,-11,3,-12,3,-12,4,-11,4,-11,4,-9,4,-8,4,-5,3,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,-3,-14,-3,-14,-3,-13,-3,-13,-3,-11,-2,-9,-1,-7,0,-5,2,-3,3,-2,5,0,7,1,8,2,10,2,11,2,11,2,12,1,11,1,10,1,9,1,8,1,6,1,4,1,2,1,0,2,-2,3,0,0,-6,6,-7,7,-8,8,-8,9,-8,9,-8,9,-8,9,-6,8,-5,7,-4,6,-3,4,-2,2,-2,0,-1,-3,-1,-5,-1,-7,-1,-9,-1,-10,-2,-12,-2,-12,-3,-13,-3,-12,-3,-11,-4,-10,-4,-8,-3,-6,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,2,-14,1,-14,1,-14,1,-13,1,-11,1,-10,1,-7,2,-5,2,-3,3,0,4,1,6,3,7,5,8,5,9,6,9,6,10,5,9,5,10,5,8,4,7,3,5,3,3,2,1,2,-1,2,-4,2,0,0,-7,3,-9,4,-10,5,-11,5,-12,6,-11,6,-10,6,-9,5,-8,5,-6,4,-4,3,-3,1,-2,-1,0,-3,1,-4,2,-6,2,-8,3,-10,3,-11,3,-12,2,-12,2,-12,1,-11,0,-10,0,-8,-1,-6,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,0,-14,0,-14,1,-13,1,-13,0,-11,0,-9,1,-7,1,-5,2,-3,4,-1,5,1,6,3,7,4,8,5,9,5,10,5,10,5,10,5,9,4,8,4,7,3,5,2,3,2,1,2,-1,2,-3,2,0,0,-8,3,-9,4,-11,4,-11,5,-11,6,-11,6,-10,6,-9,5,-8,5,-6,4,-4,2,-3,1,-1,-1,0,-3,1,-4,2,-6,3,-8,3,-9,4,-11,3,-12,3,-12,3,-12,2,-11,1,-10,0,-8,0,-6,0,0,0,0,0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0,0,0,0,0,3,-3,3,0,4,3,4,4,5,7,6,7,7,9,8,9,8,9,8,9,8,8,7,7,6,6,5,5,3,4,1,3,-1,2,-3,2,-5,2,-7,2,-9,2,-10,2,-11,4,-11,4,-11,4,-11,5,0,0,-9,4,-8,4,-6,2,-5,1,-3,0,-1,-2,0,-4,1,-6,2,-8,2,-10,2,-10,2,-11,2,-11,1,-11,1,-11,0,-10,0,-9,-1,-7,-1,-6,-1,-3,-1,-1,-1,1,0,3,1,5,2,6,2,7,0,0,0,0,0,0,0,0,0,0]']
    # test_predict(data)

    '''
    训练模型
    '''
    # test_train()