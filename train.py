#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import dataset
import torch.optim as optim
import argparse
from yolo_voc.darknet import Darknet
from utils import *

FLAGS = None
global loss_layers

def main():
    data_options = read_data_file(FLAGS.data)
    net_options = parse_cfg(FLAGS.config)[0]

    use_cuda = True if torch.cuda.is_available() else False
    device = 'cpu' # ''cuda' if use_cuda else 'cpu'

    train_data = data_options['train']
    test_data = data_options['valid']
    names = data_options['names']

    batch_size = int(net_options['batch'])
    learning_rate = float(net_options['learning_rate'])
    hue = float(net_options['hue'])
    hue = float(net_options['hue'])
    exposure = float(net_options['exposure'])
    saturation = float(net_options['saturation'])
    momentum = float(net_options['momentum'])

    epochs = 10

    model = Darknet(FLAGS.config)
    torch.manual_seed(0)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = data_options['gpus']
        torch.cuda.manual_seed(0)

    model = model.to(device)
    loss_layers = model.loss_layers
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_data = dataset.YoloDataset(train_data, (model.width, model.height),
                                     transform=transforms.ToTensor(), train=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for idx, (images, labels) in enumerate(train_loader):
            #print(idx, images.shape, labels.shape)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            org_loss = []
            for i, l in enumerate(loss_layers):
                l.seen += labels.data.size(0)
                ol = l(output[i], labels)
                org_loss.append(ol)
            sum(org_loss).backward()
            optimizer.step()


            #print(org_loss)
            #print(output[0].shape)
            #print(labels.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',
                        type=str, default='data/voc.data', help='data description info.')
    parser.add_argument('--config', '-c',
                        type=str, default='data/yolo_v3.cfg', help='cfg file.')
    parser.add_argument('--weight', '-w',
                        type=str, default='data/yolov3.weights', help='yolov3 weight file.')
    FLAGS = parser.parse_args()
    main()
