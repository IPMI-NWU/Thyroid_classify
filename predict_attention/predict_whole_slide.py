import cv2


from kfb import kfbslide
from kfb.DeepZoom import DeepZoomGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
from Rnn_attention_embeding import Encoder

# 对RNN权重进行可视化
def is_valid_patch(image):
    channel_max = np.max(image, axis=2)
    channel_min = np.min(image, axis=2)
    diff = channel_max - channel_min
    return (len(np.where(diff > 30)[0]) / (512 * 512)) > (1 / 4)


class CNNEncoder():
    def __init__(self):
        res_net = models.resnet50(pretrained=True)
        modules = list(res_net.children())[:-1]
        self.res_net = nn.Sequential(*modules)
        self.res_net.eval()
        self.res_net.cuda()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_feature(self, image):
        with torch.no_grad():
            image = Image.fromarray(image)
            image = self.transform(image)
            image.unsqueeze_(dim=0)
            image = image.cuda()
            out = self.res_net(image)
            out = out.flatten().cpu().numpy()
        return out


class RNNEncoder():
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 构建模型
        net = Encoder()

        net.load_state_dict(
            torch.load("/media/wu24/wu24/wu24/Thyroid/unsupervised/Thyroid/model/resnet152_ce/epoch99.pth")['net'])
        self.net = net.to(device)
        self.net.eval()

    def get_weight(self, cnn_features):
        with torch.no_grad():
            cnn_features = torch.from_numpy(cnn_features).unsqueeze(0).cuda()
            return self.net(cnn_features)


if __name__ == '__main__':
    src_kfb_dir = '/media/wu24/wu24/wu24/Thyroid/Training347'
    des_png_dir = '/media/wu24/wu24/wu24/Thyroid/unsupervised/data/c2'
    kfb_names = os.listdir(src_kfb_dir)
    cnn_encoder = CNNEncoder()
    rnn_encoder = RNNEncoder()

    for kfb_name in kfb_names:
        if kfb_name == '1210100.kfb':
            kfb_path = os.path.join(src_kfb_dir, kfb_name)
            slide = kfbslide.KfbSlide(kfb_path)
            slide_level = 2
            zoom_level = -slide_level - 1
            tile_size = 512

            zoom = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
            zoom_level = [i for i in range(zoom.level_count)][zoom_level]

            x_nums, y_nums = zoom.level_tiles[zoom_level]
            x_nums -= 1
            y_nums -= 1
            xs = []
            ys = []
            features = []
            image_show = np.zeros([y_nums, x_nums], dtype=np.float)
            for x in range(x_nums):
                for y in range(y_nums):
                    tile = zoom.get_tile(zoom_level, [x, y])
                    if is_valid_patch(tile):
                        xs.append(x)
                        ys.append(y)
                        features.append(cnn_encoder.get_feature(tile))
            features = np.array(features)
            weights, outputs = rnn_encoder.get_weight(features)
            weights = weights.flatten().cpu().numpy()
            for x, y, weight in zip(xs, ys, weights):
                image_show[y, x] = weight
            prop_map = image_show
            cam = (prop_map - np.min(prop_map)) / (np.max(prop_map) - np.min(prop_map))
            cam = np.uint8(255 * cam)
            heat_map = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            width, height = slide.level_dimensions[slide_level]
            img = slide.read_region((0, 0), 6, (width, height))

            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            heat_map = cv2.resize(heat_map, (width, height))
            result = np.array(heat_map * 0.8 + img * 0.2, dtype=np.uint8)
            cv2.imwrite(os.path.join(des_png_dir, kfb_name.replace('.kfb', '_our.png')), result)

