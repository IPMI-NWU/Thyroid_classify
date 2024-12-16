import torch
import torch.nn as nn
import torch.nn.functional as F


# 这个版本首先将从CNN提取到的特征进行特征将维，然后送进RNN进行训练
class Encoder(torch.nn.Module):
    def __init__(self, input_size=1664, embed_size=512, hidden_size=512 * 2, num_layers=3, num_classes=2):
        super(Encoder, self).__init__()
        # print('rnn:embed_size:{},num_layers:{}'.format(embed_size,num_layers))
        self.input_size = input_size
        self.embed_size = embed_size
        # 每个patch经过CNN编码后成为input_size维的向量，然后通过特征压缩模块将特征压缩成embed_size维
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_p = 0.5
        # 双向LSTM
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        # 注意力机制所需要的变量
        self.weight_W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_size, 1))
        # 二分类
        self.fc = nn.Linear(hidden_size, num_classes)
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, features):
        # features的尺寸为[batch,num_patch, 2048]
        # 首先将维度变化为[batch,num_patch, 128]
        # print(features.size())
        cnn_embed_seq = []
        for i in range(features.size()[1]):
            feature = features[:, i, :]
            x = self.fc1(feature)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc2(x)
            # x = F.relu(x)
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            # x = self.fc3(x)
            cnn_embed_seq.append(x)
            # print(len(cnn_embed_seq))
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # print(cnn_embed_seq.size())
        out, _ = self.lstm(cnn_embed_seq)  # out形状(batch,num_patch, hidden_size*2)
        # print(out.size())
        # 计算注意力权重
        u = torch.tanh(
            torch.matmul(out, self.weight_W))  # (batch,num_patch, hidden_size*2) (hidden_size*2,hidden_size*2)
        att = torch.matmul(u, self.weight_proj)  # (batch,num_patch, hidden_size*2) (hidden_size*2,1)
        att_score = F.softmax(att, dim=1)
        # 给予每个图像块的特征向量赋予权重
        scored_x = out * att_score
        # 把所有图像块的权重加起来形成切片级特征向量
        feat = torch.sum(scored_x, dim=1)

        y = self.fc(feat)
        return att_score, y
