import torch
from torch import nn
import torch.nn.functional as F
import math


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)


############### Transfomer #################


def a_norm(Q, K):
    """
    :param Q: Queries, shape: (batch_size, seq_length, dim_attn)
    :param K: Keys, shape: (batch_size, seq_length, dim_attn)
    :return: 返回attention矩阵，shape: (batch_size, seq_length, seq_length)
    """
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())

    return torch.softmax(m, -1)


def attention(Q, K, V):
    """
    :param Q: Queries, shape: (batch_size, seq_length, dim_attn)
    :param K: Keys, shape: (batch_size, seq_length, dim_attn)
    :param V: Values, shape: (batch_size, seq_length, dim_val)
    :return: 返回attention矩阵与V的乘积，shape: (batch_size, seq_length, dim_val)
    """

    # Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K)  # a.shape = (batch_size, seq_length, seq_length)

    return torch.matmul(a,  V)  # shape: (batch_size, seq_length, dim_val)


class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)

    def forward(self, x, kv=None):
        """
        :param x: 输入，shape: (batch_size, seq_length, dim_val)
        :param kv: 如果是decoder，kv是encoder的输出, shape: (batch_size, seq_length, dim_val)
        """

        if (kv is None):
            # Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))

        # Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))


class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        """
        :param dim_val: value的维度
        :param dim_attn: attention的维度
        :param n_heads: 多头注意力的头数
        """

        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))

        self.heads = nn.ModuleList(self.heads)  # ModuleList 将所有模块包装在一起，使其可以被识别
        # 由于多头注意力的输出是拼接在一起的，所以需要一个线性层将其合并
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        """
        :param x: 输入，shape: (batch_size, seq_length, dim_val)
        :param kv: 如果是decoder，kv是encoder的输出
        """
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))

        # combine heads，最后a.shape = (batch_size, seq_length, dim_val, n_heads)
        a = torch.stack(a, dim=-1)
        # flatten all head outputs, a.shape = (batch_size, seq_length, n_heads * dim_val)
        a = a.flatten(start_dim=2)

        x = self.fc(a)  # x.shape = (batch_size, seq_length, dim_val)

        return x


class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val

        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)
        # self.fc2 = nn.Linear(5, dim_val)

    def forward(self, x):
        """
        :param x: 输入，shape: (batch_size, seq_length, dim_input)
        :return: 返回V，shape: (batch_size, seq_length, dim_val)
        """

        x = self.fc1(x)
        return x


class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn

        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)
        # self.fc2 = nn.Linear(5, dim_attn)

    def forward(self, x):
        """
        :param x: 输入，shape: (batch_size, seq_length, dim_input)
        :return: 返回K，shape: (batch_size, seq_length, dim_attn)
        """

        x = self.fc1(x)
        return x


class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn

        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)
        # self.fc2 = nn.Linear(5, dim_attn)

    def forward(self, x):
        """
        :param x: 输入，shape: (batch_size, seq_length, dim_input)
        :return: 返回Q，shape: (batch_size, seq_length, dim_attn)
        """

        x = self.fc1(x)
        return x

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]. squeeze(1)
        return x


def get_data(batch_size, input_sequence_length, output_sequence_length):
    """
    模拟数据，如果我们调用get_data(2, 5, 3)，那么函数将返回两个形状为(2, 5, 1)和(2, 3)的张量，
    分别代表两个长度为5的输入序列和两个长度为3的输出序列。通过sigmoid函数处理，在(0,1)区间内。

    :param batch_size: batch大小
    :param input_sequence_length: 输入序列长度
    :param output_sequence_length: 输出序列长度

    :return: 返回模拟数据，
    input.shape = (batch_size, input_sequence_length, 1), 
    output.shape = (batch_size, output_sequence_length)
    """
    i = input_sequence_length + output_sequence_length

    t = torch.zeros(batch_size, 1).uniform_(0, 20 - i).int()
    # 生成一个形状为(batch_size, 1)的全零张量，然后在[0, 20 - i)范围内填充均匀分布的随机数
    # 最后转换为整数。这个张量将用于后续的偏移操作。

    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size, 1) + t
    # 首先生成一个从-10到-10+i的整数序列，然后增加一个维度并复制到batch_size行，最后加上之前生成的随机偏移t。
    # 这样得到的b张量的每一行都是一个长度为i的序列，但每行的起始值不同。shape为(batch_size, i)。

    s = torch.sigmoid(b.float())
    # 将b张量转换为浮点数，然后应用sigmoid函数。
    # sigmoid函数可以将任何实数映射到(0,1)区间，使得生成的数据更适合用于模型训练。

    # 最后，函数返回两部分数据，一部分是输入序列，另一部分是输出序列。
    # 输入序列是s的前input_sequence_length列，输出序列是s的后output_sequence_length列。
    return s[:, :input_sequence_length].unsqueeze(-1), s[:, -output_sequence_length:]
