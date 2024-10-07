import sys
import torch
import logging
from torch.nn import functional as F

from dataset import process_single_csv
from model.Transformer import Transformer
# from wiguard import config

torch.manual_seed(0)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Device: {}".format(device))

CLIP_SIZE = 100  # 截取的数据长度
SUBCARRIES = 64  # 子载波数
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

# pth_path = config['model']

model = Transformer(dim_val=DIM_VAL,
                    dim_attn=DIM_ATTN,
                    input_size=SUBCARRIES,
                    dec_seq_len=DEC_SQL_LEN,
                    out_seq_len=3,
                    n_decoder_layers=N_DECODER_LAYERS,
                    n_encoder_layers=N_ENCODER_LAYERS,
                    n_heads=N_HEADS)
model.float().to(device)
# if (not torch.cuda.is_available()):
#     model.load_state_dict(torch.load(pth_path, map_location='cpu'))
# else:
#     model.load_state_dict(torch.load(pth_path))


def test(dat_path):
    amplitude_data = process_single_csv(dat_path)

    amplitude_data = torch.tensor(amplitude_data).float().to(device)
    amplitude_data = amplitude_data.unsqueeze(0)

    output = model(amplitude_data)
    pred = F.log_softmax(output, dim=1).argmax(dim=1)
    print(pred)
    if pred[0] == 0:
        res = 'air'
    elif pred[0] == 1:
        res = 'fall'
    else:
        res = 'walk'
    print(res)
    return res


if __name__ == '__main__':
    test(sys.argv[1])
