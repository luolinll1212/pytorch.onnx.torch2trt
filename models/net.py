# -*- coding: utf-8 -*-  
import torch
import torch.nn as nn
from torchvision import models

class Resnet50(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        # 主干网络
        resnet50 = models.resnet50(pretrained=True) # 预训练
        self.features = nn.Sequential(*list(resnet50.children())[1:-2])
        # 分类
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(2048, num_classes, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = x.squeeze(2).squeeze(2)
        return x

if __name__ == '__main__':
    x = torch.randn(1,3, 32, 32)
    net = Resnet50(num_classes=10)
    out = net(x)
    print(out.size())