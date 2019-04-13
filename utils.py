#!/usr/bin/python
# encoding: utf-8

import torch
import os
import numpy as np
import math
import time
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


def parse_cfg(cfgfile):
    """
    Parse cfg file, and retur a bloc dictionary.
    :param cfgfile: cfg file path
    :return: blocks
    """
    with open(cfgfile, 'r') as f:
        lines = f.readlines()

    blocks = []     # store info of all blocks
    block = {}      # store info of single block

    for line in lines:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        if line[0] == '[':
            if block:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks


def read_data_file(datafile):
    options = dict()

    with open(datafile, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2longcpu(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def cal_iou(box1, box2):
    """
    Calculate iou between two boxes.
    :param box1: anchorbox (x,y,w,h).t()
    :param box2: anchorbox (x,y,w,h).t()
    :return: iou
    """
    w1, h1= box1[2], box1[3]
    w2, h2= box2[2], box2[3]
    x_min = min(box1[0] - w1 / 2, box2[0] - w2 / 2)
    x_max = max(box1[0] + w1 / 2, box2[0] + w2 / 2)
    y_min = min(box1[1] - h1 / 2, box2[1] - h2 / 2)
    y_max = max(box1[1] + h1 / 2, box2[1] + h2 / 2)

    all_w = x_max - x_min
    all_h = y_max - y_min

    intersec_w = w1 + w2 - all_w
    intersec_h = h1 + h2 - all_h

    intersec = intersec_h * intersec_w
    union = w1 * h1 + w2 * h2 - intersec
    if union <= 0:
        return 0.0
    iou = float(intersec / union)
    return iou


def cal_ious(boxes1, boxes2):
    """
    Calculate ious between two box sets.
    :param boxes1: attributes of boxes, (4, nA*nW*nH), (x, y, w,h)
    :param boxes2: attributes of boxes, (4, nA*nW*nH), (x, y, w,h)
    :return: ious
    """

    w1, h1 = boxes1[2], boxes1[3]
    w2, h2 = boxes2[2], boxes2[3]
    x_min = torch.min(boxes1[0] - w1 / 2, boxes2[0] - w2 / 2)
    x_max = torch.max(boxes1[0] + w1 / 2, boxes2[0] + w2 / 2)
    y_min = torch.min(boxes1[1] - h1 / 2, boxes2[1] - h2 / 2)
    y_max = torch.max(boxes1[1] + h1 / 2, boxes2[1] + h2 / 2)

    all_w = x_max - x_min
    all_h = y_max - y_min

    intersec_w = w1 + w2 - all_w
    intersec_h = h1 + h2 - all_h

    intersec = intersec_h * intersec_w
    intersec[intersec < 0] = 0
    union = w1 * h1 + w2 * h2 - intersec
    ious = intersec / union
    return ious


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/5, 5)   # to avoid single truth problem
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def read_class_names(classfile):
    with open(classfile, 'r') as f:
        lines = f.readlines()
    name_list = []
    for line in lines:
        name_list.append(line.strip())
    return name_list


def get_all_boxes(result, net_shape, conf_thresh, num_classes, device='cpu'):
    """
    combine three scale prediction boxes
    :param result: three scale prediction of yolo layers
    :param net_shape: (net.width, net.height)
    :param conf_thresh: confidence thresh
    :param device: device
    :return: predicted boxes of three scales
    """
    assert len(result) == 3
    nB = len(result[0]['output'])
    all_boxes = [[] for i in range(nB)]
    for i in range(len(result)):
        output = result[i]['output'].data
        anchors = result[i]['anchors']
        b = get_yolo_boxes(output, net_shape, anchors, conf_thresh, num_classes, device)
        for j in range(nB):
            all_boxes[j] += b[j]
    return all_boxes


def get_yolo_boxes(output, net_shape, anchors, conf_thresh=0.25, num_classes=20, device='cpu'):
    """
    one scale prediction boxes
    :param output: one scale prediction
    :param net_shape: (net.width, net.height)
    :param anchors: corresponding different scale anchors
    :param conf_thresh: confidence thresh
    :param device: device
    :return: predicted boxes of one scale
    """
    net_w, net_h = net_shape
    nC = int(num_classes)
    nB = output.shape[0]
    nA = len(anchors)
    nH = output.data.size(2)
    nW = output.data.size(3)
    anchors = torch.FloatTensor(anchors).view(nA, -1).to(device)
    cls_anchor_dim = nB * nA * nW * nH

    output = output.view(nB, nA, (5 + nC), nW, nH)
    cls_grid = torch.LongTensor(range(5, 5 + nC, 1)).to(device)
    ix = torch.LongTensor(range(5)).to(device)
    pred_boxes = torch.FloatTensor(4, cls_anchor_dim)

    # coordinate
    coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nW * nH).transpose(0, 1) \
        .contiguous().view(-1, cls_anchor_dim).to(device)
    coord[0:2] = coord[0:2].sigmoid()

    # confidence
    confs = output.index_select(2, ix[4]).view(cls_anchor_dim).sigmoid().to(device)


    # class
    cls = output.index_select(2, cls_grid).view(nB * nA, nC, nW * nH) \
        .transpose(1, 2).contiguous().view(cls_anchor_dim, nC).to(device)
    cls_conf = torch.nn.Softmax(dim=1)(cls)
    cls_max_confs, cls_max_ids = torch.max(cls_conf, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1) \
        .view(cls_anchor_dim).to(device)

    anchor_w = anchors.index_select(1, ix[0]).repeat(nB, nW * nH).view(cls_anchor_dim)
    anchor_h = anchors.index_select(1, ix[1]).repeat(nB, nW * nH).view(cls_anchor_dim)

    pred_boxes[0] = coord[0] + grid_x
    pred_boxes[1] = coord[1] + grid_y
    pred_boxes[2] = coord[2].exp() * anchor_w
    pred_boxes[3] = coord[3].exp() * anchor_h

    confs = convert2cpu(confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2longcpu(cls_max_ids)
    pred_boxes = convert2cpu(pred_boxes)
    pred_boxes = pred_boxes.transpose(0, 1).contiguous()
    # batch boxes
    all_boxes = []
    for b in range(nB):
        # for every image
        boxes = []
        for cy in range(nH):
            for cx in range(nW):
                for i in range(nA):
                    index = b * nA * nW * nH + i * nW * nH + cy * nW + cx
                    target_conf = confs[index]

                    if target_conf > conf_thresh:
                        cls_max_id = cls_max_ids[index]
                        cls_max_conf = cls_max_confs[index]
                        target_box = pred_boxes[index]
                        bx = target_box[0]
                        by = target_box[1]
                        bw = target_box[2]
                        bh = target_box[3]
                        box = [bx / nW, by / nH, bw / net_w, bh / net_h, target_conf, cls_max_conf, cls_max_id]
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes


def nms(boxes, nms_thresh):
    res = []
    # transfer tensor to numpy matrix
    for box in boxes:
        temp = []
        for item in box:
            if torch.is_tensor(item):
                item = float(item.numpy())
            temp.append(item)
        res.append(temp)
    boxes = res
    conf = np.zeros(len(boxes))
    for i in range(len(boxes)):
        conf[i] = boxes[i][4]
    sortIndex = list(reversed(np.argsort(conf)))
    out_boxes = []
    for i in range(len(sortIndex)):
        box_i = boxes[sortIndex[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(sortIndex)):
                box_j = boxes[sortIndex[j]]
                if cal_iou(box_i, box_j) > nms_thresh:
                    box_j[4] = 0
    return np.array(out_boxes)


def linear_penalty(s, iou, thresh):
    if s > thresh:
        return s*(1 - iou)
    else:
        return s


def gaussian_penalty(s, iou, sigma=0.5):
    return s * np.exp(-math.pow(iou, 2)/sigma)


def soft_nms(boxes, nms_thresh):
    res = []
    # transfer tensor to numpy matrix
    for box in boxes:
        temp = []
        for item in box:
            if torch.is_tensor(item):
                item = float(item.numpy())
            temp.append(item)
        res.append(temp)
    boxes = res
    conf = np.zeros(len(boxes))
    for i in range(len(boxes)):
        conf[i] = boxes[i][4]
    sortIndex = list(reversed(np.argsort(conf)))
    for i in range(len(sortIndex)):
        box_i = boxes[sortIndex[i]]
        for j in range(i + 1, len(sortIndex)):
            box_j = boxes[sortIndex[j]]
            iou = cal_iou(box_i, box_j)
            conf[sortIndex[j]] = linear_penalty(conf[sortIndex[j]], iou, thresh=0.4)
            # conf[sortIndex[j]] = gaussian_penalty(conf[sortIndex[j]], iou, sigma=0.5)
            box_j[4] = conf[sortIndex[j]]
    out_boxes = []
    for box in boxes:
        if box[4] > nms_thresh:
            out_boxes.append(box)
    return np.array(out_boxes)


def image2tensor(img):
    assert isinstance(img, Image.Image)
    transform = transforms.ToTensor()
    img = transform(img).unsqueeze(0)
    return img


def drawrect(drawcontext, xy, outline=None, width=0):
    x1, y1, x2, y2 = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def drawtext(img, pos, text, bgcolor=(255, 255, 255), font=None):
    if font is None:
        font = ImageFont.load_default().font
    (tw, th) = font.getsize(text)
    box_img = Image.new('RGB', (tw+2, th+2), bgcolor)
    ImageDraw.Draw(box_img).text((0, 0), text, fill=(0, 0, 0, 255), font=font)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    sx, sy = pos[0], pos[1]-th-2
    if sx < 0:
        sx = 0
    if sy < 0:
        sy = 0
    img.paste(box_img, (int(sx), int(sy)))


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd", 14)
    except Exception:
        font = None
    print("%d box(es) is(are) found" % len(boxes))
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[int(cls_id)], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            text = "{} : {:.3f}".format(class_names[int(cls_id)], cls_conf)
            drawtext(img, (x1, y1), text, bgcolor=rgb, font=font)
        drawrect(draw, [x1, y1, x2, y2], outline=rgb, width=2)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=False):
    model.eval()
    t0 = time.time()
    img = image2tensor(img)
    t1 = time.time()

    img = img.to(torch.device("cuda" if use_cuda else "cpu"))
    t2 = time.time()

    out_boxes = model(img)

    shape = (model.width, model.height)
    boxes = get_all_boxes(out_boxes, shape, conf_thresh, model.num_classes)[0]

    t3 = time.time()
    # boxes = soft_nms(boxes, nms_thresh)
    boxes = nms(boxes, nms_thresh)
    t4 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('             nms : %f' % (t4 - t3))
        print('           total : %f' % (t4 - t0))
        print('-----------------------------------')
    return boxes
