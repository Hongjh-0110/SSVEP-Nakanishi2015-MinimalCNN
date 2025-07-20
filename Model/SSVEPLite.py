# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/30 17:05
import torch
from torch import nn
from einops import rearrange
# from Utils.Constraint import Conv2dWithConstraint
import numpy as np


class SSVEPLite(nn.Module):
    def __init__(self, num_channels, T, num_classes, F1=16, D=4, kernelength=32, dropout=0.15):
        super(SSVEPLite, self).__init__()
        # print('num_channels, T, num_classes, F1, D, kernelength, dropout',num_channels, T, num_classes, F1, D, kernelength, dropout)
        self.spectral_spatial = nn.Sequential(
                # 使用 padding='same' 保持尺寸
                nn.Conv2d(1, F1, (1, kernelength), bias=False, padding='same'),
                nn.BatchNorm2d(F1),
                Conv2dWithConstraint(F1, F1*D, (num_channels, 1), bias=False, groups=F1, max_norm=1.),
                nn.BatchNorm2d(F1*D),
                nn.ELU(),
                nn.Dropout(dropout)
            )
        self.flatten = nn.Flatten()
        # 使用虚拟输入确定输出维度
        x = torch.ones((1, 1, num_channels, T))
        out = self.flatten(self.spectral_spatial(x))
        out_dim = out.shape[1]
        self.classifier_1 = LinearWithConstraint(out_dim, num_classes, max_norm=0.25)


    def forward(self, X):
        out = self.classifier_1(self.flatten(self.spectral_spatial(X)))
        return out

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *config, max_norm=1., **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1., **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)
