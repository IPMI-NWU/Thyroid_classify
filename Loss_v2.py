import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def one_hot(labels, batch_size, num_classes):
    device = labels.device
    one_hot_code = torch.zeros([batch_size, num_classes]).to(device)

    for i, index in enumerate(labels):
        one_hot_code[i, index.detach().item()] = 1

    return one_hot_code


def mse(predicts, targets):
    return torch.mean((targets - predicts) ** 2, dim=1)


def entropy(predicts, targets):
    result = -torch.log(torch.sum(predicts * targets, dim=1))
    return result


def penalty(targets, predictions, predicts):
    loss = 0
    count = 0
    for i in range(len(targets)):
        target = targets[i].item()
        prediction = predictions[i].item()
        if target == 1 and prediction == 0:
            loss += torch.log(1 / (1-predicts[i, 0]))
            count += 1

    if count != 0:
        return loss / count
    else:
        return 0


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predicts, targets):
        batch_size, num_classes = predicts.size()
        targets_one_hot = one_hot(targets, batch_size, num_classes)
        predicts = torch.softmax(predicts, dim=1)
        result = entropy(predicts, targets_one_hot)
        return torch.mean(result)


class PenalizeLoss(nn.Module):
    def __init__(self, weight):
        super(PenalizeLoss, self).__init__()
        self.weight = weight

    def forward(self, predicts, targets):
        batch_size, num_classes = predicts.size()
        targets_one_hot = one_hot(targets, batch_size, num_classes)
        predicts = torch.softmax(predicts, dim=1)
        _, predictions = predicts.max(1)
        ce_loss = torch.mean(entropy(predicts, targets_one_hot))
        penalty_loss = penalty(targets, predictions, predicts)
        return ce_loss + self.weight * penalty_loss


def get_loss(args):
    if args.loss == 'pce':
        print('use pce_v2{}'.format(args.weight))
        return PenalizeLoss(args.weight)
    elif args.loss == 'ce':
        print('use ce')
        return CrossEntropyLoss()
    elif args.loss == 'focal_loss':
        print('use focal_loss')
        return FocalLoss(class_num=2)
    else:
        raise Exception
