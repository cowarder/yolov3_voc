import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import parse_cfg


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class UpSample(nn.Module):

    def __init__(self, scale_factor=2, mode="nearest"):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        assert (x.dim() == 4)
        return F.interpolate(x, scale_factor=self.scale_factor,mode=self.mode)


class YoloLayer(nn.Module):

    def __init__(self):
        super(YoloLayer, self).__init__()



class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.modules = self.create_modules(self.blocks)

    def create_modules(self, blocks):
        self.net_info = blocks[0]
        self.width = int(self.net_info['width'])
        self.height = int(self.net_info['height'])
        prev_filters = int(self.net_info['channels'])
        out_filters = []
        models = nn.ModuleList()

        for index, block in enumerate(blocks[1:]):

            module = nn.Sequential()

            if block["type"] == 'convolutional':
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

                conv_layer = nn.Conv2d(prev_filters, filters, stride, padding, bias=bias)
                module.add_module('conv_{0}'.format(index), conv_layer)

                if batch_normalize:
                    module.add_module('batch_norm_{}'.format(index), nn.BatchNorm2d(kernel_size))

                if activation_func == 'leaky':
                    activation = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module('leaky_{0}'.format(index), activation)

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                module.add_module('upsample_{0}'.format(index), UpSample(scale_factor=stride, mode='nearest'))

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + index for i in layers]
                if len(layers) == 1:
                    filters = out_filters[layers[0]]
                else:
                    filters = out_filters[layers[0]] + out_filters[layers[1]]
                module.add_module('route_{0}'.format(index), EmptyLayer())

            elif block['type'] == 'shortcut':
                module.add_module('shortcut_{0}'.format(index), EmptyLayer())

            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer()
                anchors = block['anchors'].split(',')
                mask = block['mask'].split(',')
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.anchor_mask = [int(i) for i in mask]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                yolo_layer.ignore_thresh = float(block['ignore_thresh'])
                yolo_layer.truth_thresh = float(block['truth_thresh'])
                yolo_layer.layer_num = index
                yolo_layer.net_width = self.width
                yolo_layer.net_height = self.height
                module.add_module('yolo_{0}'.format(index), yolo_layer)

            models.append(module)
            prev_filters = filters
            out_filters.append(prev_filters)

        return models


if __name__=="__main__":
    darknet = Darknet("./data/yolo_v3.cfg")
    print(darknet.create_modules)
