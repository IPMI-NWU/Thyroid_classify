import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('/home/momo/workspace/Thyroid/unsupervised/Thyroid')  # 加入路径，确保在命令行能够调用
from DataSet import DataSet
from Loss import get_loss
from Rnn_attention_embeding import Encoder
from utils import progress_bar

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--device', default='cuda', type=str, help='')
parser.add_argument('--epochs', default=100, type=int, help='')
parser.add_argument('--loss', default='focal_loss', type=str, help='choose loss[ce|pce]')
parser.add_argument('--train_txt',
                    default='/home/momo/workspace/Thyroid/unsupervised/Thyroid/cnn_encoder/train_wsi.txt',
                    type=str, help='')
parser.add_argument('--train_dir',
                    default='/home/momo/workspace/Thyroid/unsupervised/data/cnn_feature_new/densenet169/train', type=str,
                    help='')

parser.add_argument('--val_txt', default='/home/momo/workspace/Thyroid/unsupervised/Thyroid/cnn_encoder/val_wsi.txt',
                    type=str, help='')
parser.add_argument('--val_dir', default='/home/momo/workspace/Thyroid/unsupervised/data/cnn_feature_new/densenet169/val',
                    type=str,
                    help='')
parser.add_argument('--save_dir', default='../model/densenet169_focal', type=str, help='')
args = parser.parse_args()

# 设置GPU
if args.device == 'cuda':
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
train_set = DataSet(args.train_dir, args.train_txt)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
val_set = DataSet(args.val_dir, args.val_txt)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
print('==> Building model..')
net = Encoder(input_size=1664)
net = net.to(args.device)
# criterion = nn.CrossEntropyLoss()
criterion = get_loss(args)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
best_acc = 0
start_epoch = 0
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

print(
    'parameter：cnn_model is densenet169,lr is {},epochs is {},save_dir is {}'.format(args.lr, args.epochs, args.save_dir))


# Training
def train():
    global epoch
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        _, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test():
    global best_acc, epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            _, outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc

    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'epoch{}.pth'.format(epoch))
    torch.save(state, save_path)


for epoch in range(start_epoch, start_epoch + args.epochs):
    train()
    test()
