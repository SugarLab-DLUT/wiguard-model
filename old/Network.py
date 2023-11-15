import torch.nn as nn
import torch

class  MyLSTM(nn.Module): #out_size=10 代表有十种动作，应根据实际情况更改. in_size=10 代表降维后数据长度为10, h_size初始为20
    def __init__(self, in_size=10, h_size=128, layers=2, out_size=10, batch_first=True, seq_len=30, bidirection=True, batch_size=8):
        super(MyLSTM,self).__init__()
        num_directions = 2 if bidirection else 1
        self.ch = seq_len
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=h_size, num_layers=layers, batch_first=batch_first, bidirectional=bidirection)
        self.resConv = nn.Sequential(
            nn.Conv1d(in_channels=self.ch, out_channels=self.ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=self.ch, out_channels=self.ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=self.ch, out_channels=self.ch, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1) #tiem step attention
            #nn.Tanh()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=seq_len * h_size * num_directions*2, out_features=512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=out_size),
            nn.Dropout(p=0.5),
            nn.Softmax(dim=1)
        )
        self.mainConv = nn.Sequential(
            nn.Conv1d(in_channels=self.ch, out_channels=self.ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x, _ = self.lstm(x) #x.shape = (batch_size, 30, 40)
        main_path_x = self.mainConv(x)#主路径
        res_path_x = self.resConv(x)#旁路径
        #print(main_path_x.shape, res_path_x.shape)
        attention_x = torch.mul(main_path_x, res_path_x)#自注意力机制，两路径输出矩阵的对应位置相乘得到attention输出
        ## x = x + attention_x
        x = torch.cat((main_path_x, attention_x), dim=2)#attention的输出与LSTM的输出堆叠在一起
        x = x.contiguous().view(x.shape[0], -1)#改成二维：（batch_size，feature_sizes）才能输入fc层
        out = self.fc(x)#分类并做softmax
        
        return out

if __name__ == "__main__":
    model = MyLSTM(in_size=30, seq_len=120, batch_size=64)
    x = torch.randn(64, 120, 30)   #batch_size=64,seq_len=30,input_size=10,因为降维后的数据是30*10的，
    y = model(x)                #batch_size不用更改，若降维后的数据变化，input_size要相应更改
    print(y.shape)                  #seq_len=30 代表30个载波,不用更改

    print(model) #查看模型具体结构
    
 
