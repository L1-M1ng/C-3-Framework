import os.path

from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *

# print(os.path.dirname(__file__))
model_path = 'C:/Users/LIMING/LM/Project/Crowd_Counting/C-3-Framework/models/PyTorch_Pretrained/vgg16-397923af.pth'
## 相对路径 'PyTorch_Pretrained/vgg16-397923af.pth'


class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        # vgg = models.vgg16(pretrained=pretrained)

        vgg = models.vgg16(pretrained=False) #使用已经下载好的预训练模型
        if pretrained:
            # print(os.path.dirname(__file__))
            vgg.load_state_dict(torch.load(model_path))

        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])

        self.de_pred = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))



    def forward(self, x):
        x = self.features4(x)       
        x = self.de_pred(x)

        x = F.upsample(x,scale_factor=8)

        return x