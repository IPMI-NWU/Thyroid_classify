import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

sys.path.append('/media/wu24/wu24/wu24/Thyroid/unsupervised/Thyroid')
from Rnn_attention_embeding import Encoder
from DataSet import DataSet



def predict():
    net = Encoder()
    state = torch.load(model_path)
    net.load_state_dict(state['net'])
    result = np.array([[0, 0], [0, 0]])
    if state['acc']>=0:
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
            print(i)
            print( state['acc'])
            print(confusion_matrix(labels,predicts))






if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载测试数据
    print('==> Preparing data..')
    test_txt = '/media/wu24/wu24/wu24/Thyroid/unsupervised/Thyroid/cnn_encoder/test_wsi.txt'
    test_dir = '/media/wu24/wu24/wu24/Thyroid/unsupervised/data/cnn_feature/densenet169/test'
    test_set = DataSet(test_dir, test_txt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
    model_dir = '/media/wu24/wu24/wu24/Thyroid/unsupervised/Thyroid/model/densenet169_pce10.0'

    # max_acc = 0
    # for i in range(100):
    #     model_path = os.path.join(model_dir, 'epoch{}.pth'.format(i))
    #     state = torch.load(model_path)
    #     max_acc = max(max_acc, state['acc'])
    for i in range(0,100):
        model_path = os.path.join(model_dir, 'epoch{}.pth'.format(i))
        predict()

