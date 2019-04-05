#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
import math
import sys
# from torchsummary import summary


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
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YoloLayer(nn.Module):

    def __init__(self, num_classes=20, anchors=[], use_cuda=None):
        super(YoloLayer, self).__init__()

        self.num_classes = num_classes                      # 类别数目
        self.anchors = anchors                              # 分解为单个元素的anchor坐标
        self.layer_num = 0                                  # 表示在网络中是属于第几层
        self.ignore_thresh = 0.5                            # IOU大于这个thresh的box将被计算loss
        self.truth_thresh = 1.0
        self.randomd = 1                                    # 是否开启多尺度训练
        self.seen = 0
        self.net_width = 416
        self.net_width = 416
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda else 'cpu'
        self.nth_layer = 0

    def build_targets(self, pred_boxes, target, anchors, nA, nH, nW):
        """

        :param pred_boxes: (nB*nA*nH*nW, 4)
        :param target: (batch_size, 250)
        :param anchors: (nA, 2)
        :param nA:
        :param nH:
        :param nW:
        :return:
        """
        nB = target.size(0)                                         # batch size
        noobj_mask = torch.ones(nB, nA, nH, nW)
        obj_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW, self.num_classes)

        nAnchors = nA * nH * nW
        nPixels = nH * nW
        nGT = 0
        nRecall = 0
        nRecall75 = 0

        anchors = anchors.to("cpu")

        # for every image
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()    # (4, nA*nH*nW)
            cur_ious = torch.zeros(nAnchors)
            # tbox(1*250)
            tbox = target[b].view(-1, 5).to("cpu")              # 一张图片里面的所有box(c,x,y,w,h),50*5

            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH       # 根据网格数目对结果进行扩大
                gw, gh = tbox[t][3] * self.net_width, tbox[t][4] * self.net_height
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                #   cur_pred_boxes(4, nA*nH*nW)          cur_gt_boxes(4, nA*nH*nW)
                # 所有的额prediction与所有ground truth计算iou值，保留每个prediction最大的iou值
                cur_ious = torch.max(cur_ious, cal_ious(cur_pred_boxes, cur_gt_boxes))
            # 过滤掉cur_pre_boxes中与所有ground truth的iou值均低于threshold的预测box(noobj_mask = 0)
            ignore_ix = (cur_ious > self.ignore_thresh).view(nA, nH, nW)
            noobj_mask[b][ignore_ix] = 0

            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1                                        # ground truth
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * self.net_width, tbox[t][4] * self.net_height
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)

                # 只利用了ground trhth的边长信息来计算iou值，从而找出跟当前ground truth最匹配的anchor(1 of 3)
                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()           # 4*3
                anchor_boxes = torch.cat((torch.zeros(nA, len(anchors)), anchors), 1).t()    # 4*3
                # best_n表示3个anchor中第几个anchor最优
                _, best_n = torch.max(cal_ious(anchor_boxes, tmp_gt_boxes), 0)

                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                # 一个cell只有一个box负责预测
                pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
                iou = cal_iou(gt_box, pred_box)

                obj_mask[b][best_n][gj][gi] = 1
                noobj_mask[b][best_n][gj][gi] = 0
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw/anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh/anchors[best_n][1])
                tcls[b][best_n][gj][gi][int(tbox[t][0])] = 1
                tconf[b][best_n][gj][gi] = iou

                if iou > 0.5:
                    nRecall += 1
                    if iou > 0.75:
                        nRecall75 += 1

        return nGT, nRecall, nRecall75, obj_mask, noobj_mask, tcoord, tconf, tcls

    def forward(self, output, target):
        nB = output.data.size(0)                            # batch size
        nA = len(self.anchors)                              # anchor num
        nC = self.num_classes                               # class num
        nH = output.data.size(2)                            # grid num along y axis
        nW = output.data.size(3)                            # grid num along x axis
        anchors = torch.FloatTensor(self.anchors).view(nA, -1).to(self.device)
        cls_anchor_dim = nB * nA * nH * nW

        output = output.view(nB, nA, (5+nC), nH, nW)
        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long().to(self.device)     # 5,6,7...5+nC-1
        ix = torch.LongTensor(range(5)).to(self.device)
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim).to(self.device)       # 用于保存每个anchor box的(x,y,w,h)坐标信息

        # 获取每个anchor的坐标
        # index_select(2, ix[0:4])，其中2表示维度，ix[0:4]表示索引，只选择了坐标的四列
        coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).transpose(0, 1) \
            .contiguous().view(-1, cls_anchor_dim)  # 每一个anchor的x, y, w, h
        coord[0:2] = coord[0:2].sigmoid()  # 前两行表示坐标，后两行表示长宽

        # 获取每个anchor的置信度
        conf = output.index_select(2, ix[4]).view(cls_anchor_dim).sigmoid()     # 置信度

        # 获取每个anchor的class
        cls = output.index_select(2, cls_grid)
        cls = cls.view(nB*nA, nC, nH*nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC).to(self.device)

        # 获取网格的(x, y),0~nW-1,0~nH-1
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1) \
            .view(cls_anchor_dim).to(self.device)

        # 获取anchor信息
        anchor_w = anchors.index_select(1, ix[0]).repeat(nB, nH * nW).view(cls_anchor_dim)
        anchor_h = anchors.index_select(1, ix[1]).repeat(nB, nH * nW).view(cls_anchor_dim)

        # coord 与 pred_boxes形状一致 yolov3(2.1)
        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h

        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4)).detach()

        # pred_boxes(nB*nA*nH*nW, 4)   target(batchsize, 250)其中250是50*5    anchors(nA, 2)
        nGT, nRecall, nRecall75, obj_mask, noobj_mask, tcoord, tconf, tcls = \
            self.build_targets(pred_boxes, target.detach(), anchors.detach(), nA, nH, nW)

        conf_mask = (obj_mask + noobj_mask).view(cls_anchor_dim).to(self.device)
        obj_mask = (obj_mask == 1).view(cls_anchor_dim)

        nProposals = int((conf > 0.25).sum())

        coord = coord[:, obj_mask]
        tcoord = tcoord.view(4, cls_anchor_dim)[:, obj_mask].to(self.device)

        tconf = tconf.view(cls_anchor_dim).to(self.device)

        cls = cls[obj_mask, :].to(self.device)
        tcls = tcls.view(cls_anchor_dim, nC)[obj_mask, :].to(self.device)

        loss_coord = nn.BCELoss(reduction='sum')(coord[0:2], tcoord[0:2]) / nB + \
                     nn.MSELoss(reduction='sum')(coord[2:4], tcoord[2:4]) / nB
        loss_conf = nn.BCELoss(reduction='sum')(conf * conf_mask, tconf * conf_mask) / nB
        loss_cls = nn.BCEWithLogitsLoss(reduction='sum')(cls, tcls) / nB

        loss = loss_coord + loss_conf + loss_cls

        print('%d: Layer(%03d) nGT %3d, nRC %3d, nRC75 %3d, nPP %3d, loss: box %6.3f,'
              ' conf %6.3f, class %6.3f, total %7.3f'
              % (self.seen, self.nth_layer, nGT, nRecall, nRecall75, nProposals, loss_coord, loss_conf, loss_cls, loss))
        if math.isnan(loss.item()):
            print(np.any(np.isnan(conf.detach().numpy())))
            print(np.any(np.isnan(output.detach().numpy())))
            print(np.any(np.isnan(tconf.detach().numpy())))

            # print(coord, conf, tconf)
            # print(conf)

            sys.exit(0)
        return loss

    def get_mask_boxes(self, x):
        return x


class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_modules(self.blocks)
        self.loss_layers = self.get_loss_layers()
        self.seen = 0

    def get_loss_layers(self):
        loss_layers = []
        for module in self.models:
            if isinstance(module[0], YoloLayer):
                loss_layers.append(module[0])
        return loss_layers

    def forward(self, x):
        output = {}
        boxes = {}
        box_no = 0
        for index, block in enumerate(self.blocks[1:]):
            if block['type'] == 'convolutional' or block['type'] == 'upsample':
                x = self.models[index](x)
                output[index] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + index for i in layers]
                if len(layers) == 1:
                    x = output[layers[0]]
                else:
                    x1 = output[layers[0]]
                    x2 = output[layers[1]]
                    x = torch.cat([x1, x2], 1)
                output[index] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                from_layer = int(from_layer) if int(from_layer) > 0 else int(from_layer) + index
                activation = block['activation']
                x1 = output[from_layer]
                x2 = output[index - 1]
                x = x1 + x2
                output[index] = x
            elif block['type'] == 'yolo':
                boxes[box_no] = self.models[index][0].get_mask_boxes(x)
                box_no += 1
                output[index] = None

            else:
                print("Unknown type {0}".format(block['type']))
        return x if len(boxes) == 0 else boxes

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

                conv_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
                module.add_module('conv_{0}'.format(index), conv_layer)

                if batch_normalize:
                    module.add_module('batch_norm_{}'.format(index), nn.BatchNorm2d(filters))

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
                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]
                anchors = block['anchors'].split(',')
                anchors = [float(x) for x in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                yolo_layer.anchors = anchors
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.ignore_thresh = float(block['ignore_thresh'])
                yolo_layer.truth_thresh = float(block['truth_thresh'])
                yolo_layer.layer_num = index
                yolo_layer.net_width = self.width
                yolo_layer.net_height = self.height
                yolo_layer.nth_layer = index
                #module.add_module('yolo_{0}'.format(index), yolo_layer)
                module.add_module('yolo_{0}'.format(index), yolo_layer)

            models.append(module)
            prev_filters = filters
            out_filters.append(prev_filters)

        return models


if __name__=="__main__":
    darknet = Darknet("./data/yolo_v3.cfg")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    darknet = darknet.to(device)
    #summary(darknet, (3, 416, 416))
