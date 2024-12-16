import sys

sys.path.append('/home/wu24/python/project/SlideClassification')
import torch
from Rnn_attention_embeding import Encoder
from DataSet import DataSet
from utils import progress_bar

# 计算模型在测试集上的准确率

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 加载训练数据
print('==> Preparing data..')
test_txt = '/home/momo/workspace/Thyroid/unsupervised/Thyroid/cnn_encoder/test_wsi.txt'
test_dir = '/home/momo/workspace/Thyroid/unsupervised/data/cnn_feature_new/inceptionv3/test'
val_set = DataSet(test_dir, test_txt)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
# 构建模型
print('==> Building model..')
net = Encoder(input_size=2048)
net.load_state_dict(torch.load("/home/momo/workspace/Thyroid/unsupervised/Thyroid/model/inceptionv3_pce_new/epoch99.pth")['net'])
net = net.to(device)
# 开始预测
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        _, outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(val_loader), 'Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
