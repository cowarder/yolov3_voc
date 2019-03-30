import torch


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


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def iou(box1, box2):
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

def multi_ious(boxes1, boxes2):
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
