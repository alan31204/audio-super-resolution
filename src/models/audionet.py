import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import math
# import copy


# TODO: set up the audio net model
# Pytorch Official ResNet sample model
# read Wavenet paper, and understand its structures

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Convolution 3x3
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)


class AudioNet(nn.Module):
# Generic PyTorch model training code
    # def __init__(self, from_ckpt=False, n_dim=None, r=2, opt_params=default_opt):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(AudioNet, self).__init__()
        self.dconv1 = DLayer(128, 128, 65, stride=1, 32)
        self.dconv2 = DLayer(256, 256, 33, stride=1, 112)
        self.dconv3 = DLayer(512, 512, 17, stride=1, 248)
        self.dconv4 = DLayer(512, 512, 9, stride=1, 252)
        self.dconv5 = DLayer(512, 512, 9, stride=1, 252)
        self.dconv6 = DLayer(512, 512, 9, stride=1, 252)
        self.dconv7 = DLayer(512, 512, 9, stride=1, 252)
        self.dconv8 = DLayer(512, 512, 9, stride=1, 252)
        self.bneck1 = Bottleneck(512, 512, stride=1)
        self.uconv1 = ULayer()
        self.uconv2 = ULayer()
        self.uconv3 = ULayer()
        self.uconv4 = ULayer()
        self.uconv5 = ULayer()
        self.uconv6 = ULayer()
        self.uconv7 = ULayer()
        self.uconv8 = ULayer()
        self.fconv  = FinalConv(2, 2, stride=1)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # maxpool ?
    #   self.in_channels = 16
        

    # perform the usual initialization
    # self.r = r
    # Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
    #                opt_params=opt_params, log_prefix=log_prefix)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # Forward Downsample pass
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        x = self.dconv5(x)
        x = self.dconv6(x)
        x = self.dconv7(x)
        x = self.dconv8(x)

        x = self.bneck1(x)
        

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fconv(x)

        return x



# ResNet Downsample Basic Block model
class DLayer(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, padding, downsample=None):
        super(DLayer, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False) # padding not sure
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.2)
        self.downsample = downsample
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.relu(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        # not sure
        out += residual
        out = self.relu(out)

        return out


# ResNet Upsample Basic Block model
class ULayer(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, padding upsample=None):
        super(ULayer, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle()

        self.upsample = upsample
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.drop(x)
        out = self.relu(out)
        out = self.pixel_shuffle(out)
        # Stacking
        
        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        # out = self.relu(out)

        return out


# ResNet Bottleneck architecture
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=9, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.LeakyReLU(0.2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.drop(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FinalConv(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FinalConv, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=9, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle()
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.pixel_shuffle(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out





