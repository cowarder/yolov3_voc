import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import *


class FCLayer(nn.Module):

    def __init__(self, out_dim):
        super(FCLayer, self).__init__()
        self.out_dim = out_dim

    def forward(self, x):
        assert x.data.dim() == 2
        N = x.data.size(0)
        dim = x.data.size(1)
        fc = nn.Linear(dim, self.out_dim)
        x = fc(x)
        return x


class DetectionLayer(nn.Module):
    def __init__(self, classes=20, side=7, num=3, jitter=.2, object_scale=1,
                 noobject_scale=.5, class_scale=1, coord_scale=5):
        super(DetectionLayer, self).__init__()

    def forward(self, input):
        return input

    def get_input(self, x):
        return x


class Darknet19(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet19, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_modules(self.blocks)

    def create_modules(self, blocks):
        self.net_info = blocks[0]
        self.width = int(self.net_info['width'])
        self.height = int(self.net_info['height'])
        models = nn.ModuleList()
        prev_filters = int(self.net_info['channels'])

        for index, block in enumerate(self.blocks[1:]):

            module = nn.Sequential()
            if block['type'] == 'convolutional' or block['type'] == 'local':
                activation_func = block['activation']
                kernel_size = int(block['size'])
                pad = int(block['pad'])
                filters = int(block['filters'])
                stride = int(block['stride'])

                try:
                    batch_normalize = int(block['batch_normalize'])
                    bias = False
                except KeyError:
                    batch_normalize = 0
                    bias = True

                if pad:
                    padding = (kernel_size - 1) // 2
                else:
                    padding = 0

                conv_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
                module.add_module('conv_{0}'.format(index), conv_layer)

                if batch_normalize:
                    module.add_module('batch_norm_{}'.format(index), nn.BatchNorm2d(filters))

                if activation_func == 'leaky':
                    activation = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module('leaky_{0}'.format(index), activation)
                prev_filters = filters
            elif block['type'] == 'maxpool':
                size = int(block['size'])
                stride = int(block['stride'])
                maxpool = nn.MaxPool2d(size, stride)
                module.add_module("maxpool_{0}".format(index), maxpool)
            elif block['type'] == 'dropout':
                prob = float(block['probability'])
                dropout = nn.Dropout(prob)
                module.add_module('dropout_{0}'.format(index), dropout)
            elif block['type'] == 'connected':
                out_dim = int(block['output'])
                fc = FCLayer(out_dim)
                module.add_module('connected_{0}'.format(index), fc)
            elif block['type'] == 'detection':
                self.classes = int(block['classes'])
                self.coords = int(block['coords'])
                self.rescore = int(block['rescore'])
                self.side = int(block['side'])
                self.num = int(block['num'])

                self.jitter = float(block['jitter'])

                self.object_scale = float(block['object_scale'])
                self.noobject_scale = float(block['noobject_scale'])
                self.class_scale = float(block['class_scale'])
                self.coord_scale = float(block['coord_scale'])

                detection_layer = DetectionLayer(self.classes, self.side, self.num, self.jitter,
                                                 object_scale=self.object_scale, noobject_scale=self.noobject_scale,
                                                 class_scale=self.class_scale, coord_scale=self.class_scale)
                module.add_module('detection_{}'.format(index), detection_layer)
            models.append(module)
        return models

    def forward(self, x):
        for index, block in enumerate(self.blocks[1:]):
            if block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'dropout' or block['type'] == 'connected':
                x = self.models[index](x)
            elif block['type'] == 'local':
                x = self.models[index](x)
                x = x.view(x.size(0), -1)
            elif block['type'] == 'detection':
                x = self.models[index][0].get_input(x)

        return x


input = torch.randn(1, 3, 448, 448)
model = Darknet19('data/yolo.cfg')
output = model(input)
print(output.shape)