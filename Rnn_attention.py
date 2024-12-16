import torch
import torch.nn as nn
import torch.nn.functional as F


# 双向lstm+attention
# 这个版本将提取到的特征直接送进RNN网络
class Encoder(torch.nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048 * 2, num_layers=3, num_classes=2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 双向GRU，//操作为了与后面的Attention操作维度匹配，hidden_dim要取偶数！
        self.bigru = nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                             bidirectional=True, batch_first=True)
        # 由nn.Parameter定义的变量都为requires_grad=True状态
        self.weight_W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_size, 1))
        # 二分类
        self.fc = nn.Linear(hidden_size, num_classes)
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, sentence):
        gru_out, _ = self.bigru(sentence)
        # # # Attention过程，与上图中三个公式对应
        u = torch.tanh(torch.matmul(gru_out, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        # print(att_score)
        scored_x = gru_out * att_score
        # # # Attention过程结束
        feat = torch.sum(scored_x, dim=1)
        y = self.fc(feat)
        return att_score,y


