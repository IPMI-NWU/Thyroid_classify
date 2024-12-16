import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append('/media/wu24/wu24/wu24/Thyroid/unsupervised/Thyroid')
from Rnn_attention_embeding import Encoder
from DataSet import DataSet



def predict():
    net = Encoder()
    state = torch.load(model_path)
    net.load_state_dict(state['net'])
    result = np.array([[0, 0], [0, 0]])
    if state['acc']>=97:
        net = net.to(device)
        net.eval()
        correct = 0
        total = 0
        predicts = []
        labels = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                _, outputs = net(inputs)
                _, predictions = outputs.to('cpu').max(1)
                total += targets.size(0)
                correct += predictions.eq(targets).sum().item()
                result[predictions[0], targets[0]] += 1
                labels.append(targets[0])
                predicts.append(predictions[0])


            writer.write('epoch{} '.format(i))
            writer.write('acc:{}'.format(100. * correct / total))
            acc = accuracy_score(labels, predicts)
            pre = precision_score(labels, predicts)
            recall = recall_score(labels, predicts)
            f1 = f1_score(labels, predicts)
            writer.write('|{},{},{},{}|'.format(acc, pre, recall, f1))
            print(acc, pre, recall, f1)
            writer.write('00:{},01:{},10:{},11{}\n'.format(result[0, 0], result[0, 1], result[1, 0], result[1, 1]))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载测试数据
    print('==> Preparing data..')
    test_txt = '/media/wu24/wu24/wu24/Thyroid/unsupervised/Thyroid/cnn_encoder/test_wsi.txt'
    test_dir = '/media/wu24/wu24/wu24/Thyroid/unsupervised/data/cnn_feature/densenet169/test'
    test_set = DataSet(test_dir, test_txt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
    model_dir = '/media/wu24/wu24/wu24/Thyroid/unsupervised/Thyroid/model/densenet169_ce'
    writer = open('./densenet_ce.txt', 'w')
    # max_acc = 0
    # for i in range(100):
    #     model_path = os.path.join(model_dir, 'epoch{}.pth'.format(i))
    #     state = torch.load(model_path)
    #     max_acc = max(max_acc, state['acc'])
    for i in range(100):
        model_path = os.path.join(model_dir, 'epoch{}.pth'.format(i))
        predict()
    writer.close()
