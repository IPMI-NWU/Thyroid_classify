import os

import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
# 加载模型
dense_net = models.densenet169(pretrained=True)

modules = list(dense_net.children())[:-1]
dense_net = nn.Sequential(*modules)
dense_net.eval()
dense_net.cuda()
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
src_dir = '/home/momo/workspace/Thyroid/unsupervised/clean_data_new/image_all/test'
des_dir = '/home/momo/workspace/Thyroid/unsupervised/data/cnn_feature_new/densenet169/test'
patients = os.listdir(src_dir)
with torch.no_grad():
    for patient_name in patients:
        # 为每个患者生成编码矩阵
        src_patient_path = os.path.join(src_dir, patient_name)
        des_patient_path = os.path.join(des_dir, '{}.npy'.format(patient_name))
        image_names = os.listdir(src_patient_path)
        features = []
        for i in range(len(image_names)):
            # 为每张图片生成编码向量
            image_path = os.path.join(src_patient_path, '{}.png'.format(i))
            image = Image.open(image_path)
            image = transform(image)
            image.unsqueeze_(dim=0)
            image = image.cuda()
            out = dense_net(image)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out=out.flatten().cpu().numpy()
            features.append(out)
        np.save(des_patient_path, np.array(features))

