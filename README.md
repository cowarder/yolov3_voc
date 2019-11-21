# Yolov3-Pytorch
This repo is the implementation  of my graduation design.

I rewrote the code of Yolov3 and achieved the performance mentioned  in [this paper](https://arxiv.org/pdf/1804.02767.pdf)  .

In addition, I proposed an improved [NMS](https://en.wikipedia.org/wiki/Canny_edge_detector#Non-maximum_suppression) algorithm that adjust the threshold based on image information and validated it on Pascal Voc 2007 dataset using the Yolov3 model.

[data](https://github.com/cowarder/yolov3_voc/tree/master/data): Firstly, you need to prepare your Pascal VOC 2007 dataset in this folder, and you need to change the configuration in data/voc.data. Please read more in data/Data Description.txt

[Darknet.py](https://github.com/cowarder/yolov3_voc/blob/master/Darknet.py): Darknet model file.

[camera.py](https://github.com/cowarder/yolov3_voc/blob/master/camera.py): video demo, real-time implementation of trained YOLOV3 model.

[detect.py](https://github.com/cowarder/yolov3_voc/blob/master/detect.py): image demo. You can make detection of an image by this file.

[train.py](https://github.com/cowarder/yolov3_voc/blob/master/train.py): training model file.

[utils.py](https://github.com/cowarder/yolov3_voc/blob/master/utils.py): this file includes necessary utility functions.

[test.py](https://github.com/cowarder/yolov3_voc/blob/master/test.py): my algorithm experiments(this file doesn't relate to YOLOV3 training, ignore it).

[valid.py](https://github.com/cowarder/yolov3_voc/blob/master/valid.py): validation of my algorithm(this file doesn't relate to YOLOV3 training, ignore it).


