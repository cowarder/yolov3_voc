import torch
from torchvision import transforms
import sys
import os
import shutil
import time
from torch.utils.data import DataLoader, Dataset
from dataset import YoloDataset
from darknet import Darknet
from utils import read_data_file, read_class_names, get_all_boxes, correct_yolo_boxes, nms


def valid(datafile, cfgfile, weightfile, prefix='result'):
    model = Darknet(cfgfile)
    options = read_data_file(datafile)
    data_root = options['valid']
    class_root = options['names']
    names = read_class_names(class_root)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(data_root, 'r') as f:
        lines = f.readlines()
        valid_files = [item.strip() for item in lines]

    model.load_weights(weightfile)
    model = model.to(device)
    model.eval()

    data = YoloDataset(data_root,
                       shape=(model.width, model.height),
                       transform=transforms.Compose([transforms.ToTensor()]),
                       train=False)
    batch_size = 2
    kwargs = {'num_workers': 4, 'pin_memory': True}
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)

    fs = [None] * model.num_classes
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for i in range(model.num_classes):
        filename = prefix + '/' + str(names[i]) + '.txt'
        fs[i] = open(filename, 'w')
    net_shape = (model.width, model.height)

    conf_thresh = 0.005
    nms_thresh = 0.45

    fileIndex = 0
    for index, (imgs, labels, org_w, org_h) in enumerate(data_loader):
        imgs = imgs.to(device)
        output = model(imgs)

        batch_boxes = get_all_boxes(output, net_shape, conf_thresh, model.num_classes, device, validation=True)

        for i in range(len(batch_boxes)):
            fileId = os.path.basename(valid_files[fileIndex]).split('.')[0]    # gei naive image name without suffix
            w, h = float(org_w[i]), float(org_h[i])
            # print(valid_files[fileIndex], '{}/{}'.format(fileIndex+1, len(data_loader) * batch_size))
            fileIndex += 1
            boxes = batch_boxes[i]
            correct_yolo_boxes(boxes, w, h, model.width, model.height)
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2] / 2.0) * w
                y1 = (box[1] - box[3] / 2.0) * h
                x2 = (box[0] + box[2] / 2.0) * w
                y2 = (box[1] + box[3] / 2.0) * h

                # 包含物体的概率，乘以每一类的概率
                det_conf = box[4]
                for j in range((len(box)-5)//2):
                    cls_conf = box[5+2*j]
                    cls_id = int(box[5+2*j+1])
                    prob = det_conf * cls_conf
                    fs[cls_id].write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.format(fileId, prob, x1, y1, x2, y2))

    for i in range(len(fs)):
        fs[i].close()


def main():
    datafile = 'data/voc.data'
    cfgfile = 'data/yolo_v3.cfg'
    weightfile = 'model.weights'

    if not os.path.exists('result/'):
        pass
    else:
        shutil.rmtree('result/')
    os.makedirs('result')

    # vallid one model
    # valid(datafile, cfgfile, weightfile)

    # valid all models
    r = list(range(-1, 100, 10))
    r[0] = 0
    for i in r:
        print('validating epoch_{}.'.format(i))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        prefix = 'result/epoch_' + str(i)
        weightfile = 'models/epoch_' + str(i) + '.weights'
        valid(datafile, cfgfile, weightfile, prefix)
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        print('validating epoch_{} done.'.format(i))


if __name__ == '__main__':
    main()
    # %run valid.py data/voc.data data/yolo_v3.cfg data/model.weights
