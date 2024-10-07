import torch
import torch.nn as nn
import torch.nn.functional as F
from wiguard.model.utils import *



class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        """
        :param dim_val: value的维度
        :param dim_attn: attention的维度
        :param n_heads: 多头注意力的头数
        """

        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)  # 残差连接

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)

        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        """
        :param dim_val: value的维度
        :param dim_attn: attention的维度
        :param n_heads: 多头注意力的头数
        """

        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

    def forward(self, x, enc):
        """
        :param x: decoder的输入
        :param enc: encoder的输出
        """

        a = self.attn1(x).float()
        x = self.norm1(a + x).float()

        a = self.attn2(x, kv=enc).float()
        x = self.norm2(a + x).float()

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm3(x + a).float()
        return x


class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1, n_heads=1):
        """
        :param dim_val: value的维度
        :param dim_attn: attention的维度
        :param input_size: 输入的维度
        :param dec_seq_len: decoder的序列长度
        :param out_seq_len: 输出的序列长度
        :param n_decoder_layers: decoder的层数
        :param n_encoder_layers: encoder的层数
        :param n_heads: 多头注意力的头数
        """

        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len

        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads))

        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads))

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)

    def forward(self, x):
        """
        :param x: 输入的序列, shape: (batch_size, seq_len, input_size)
        """
        # encoder
        # e.shape: (batch_size, seq_len, dim_val)
        e = self.encs[0](self.pos(self.enc_input_fc(x.float())))
        # print("ecoder init")
        for enc in self.encs[1:]:
            e = enc(e)

        # decoder
        # d.shape: (batch_size, dec_seq_len, dim_val)
        d = self.decs[0](self.dec_input_fc(x[:, -self.dec_seq_len:].float()), e)
        # print(x[:,-self.dec_seq_len:].shape)  # (batch_size, dec_seq_len, input_size)
        # x[:,-self.dec_seq_len:]取出了输入序列的最后dec_seq_len个时序数据，作为decoder的输入

        for dec in self.decs[1:]:
            d = dec(d, e)

        # output
        # x.shape: (batch_size, out_seq_len)
        x = self.out_fc(d.flatten(start_dim=1))
        # flatten 的作用是将多维的输入展平，start_dim是从第几维开始展平，这里是从第1维开始展平，
        # 即将第2维到最后一维展平，得到的d.shape是(batch_size, dec_seq_len * dim_val)

        return x
