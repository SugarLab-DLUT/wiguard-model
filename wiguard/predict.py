from json import load
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import torch
import logging
from torch.nn import functional as F
from wiguard.dataset import process_single_csv_file,  CSIDataset, LABELS
from wiguard.model.Transformer import Transformer
from wiguard.process import get_amplitude
# from wiguard import config

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

load_dotenv()
torch.manual_seed(0)
model_path = os.getenv("MODEL_PATH") or "./models/model0.pth"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
    model.load_state_dict(torch.load(
        model_path, weights_only=True, map_location='cpu'))
else:
    model.load_state_dict(torch.load(model_path, weights_only=True))


def predict(csi: pd.DataFrame):
    amplitude = get_amplitude(csi)
    amplitude = torch.tensor(amplitude).float().to(device)
    amplitude = amplitude.unsqueeze(0)

    output = model(amplitude)
    pred = F.log_softmax(output, dim=1).argmax(dim=1)
    return int(pred[0].item())


if __name__ == '__main__':
    csi = pd.read_csv(sys.argv[1], header=None)
    result = predict(csi)
    print(f"Result: {LABELS[result]}")
