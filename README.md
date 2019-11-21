# Yolov3-Pytorch
This repo is the implementation  of my graduation design.

I rewrote the code of Yolov3 and achieved the performance mentioned  in [this paper](https://arxiv.org/pdf/1804.02767.pdf).  

In addition, I proposed an improved [NMS](https://en.wikipedia.org/wiki/Canny_edge_detector#Non-maximum_suppression) algorithm that adjust the threshold based on image information and validated it on Pascal VOC 2007 dataset.

* [data](https://github.com/cowarder/yolov3_voc/tree/master/data): Firstly, you need to prepare your Pascal VOC 2007 dataset in this folder, and you need to change the configuration in data/voc.data. Please read more in data/Data Description.txt.  

* [Darknet.py](https://github.com/cowarder/yolov3_voc/blob/master/Darknet.py): Darknet model file.  

* [camera.py](https://github.com/cowarder/yolov3_voc/blob/master/camera.py): video demo, real-time implementation of trained YOLOV3 model.  

* [gui.py](https://github.com/cowarder/yolov3_voc/blob/master/gui.py): graphical user interface to detect image, you can choose any image you like.  

* [detect.py](https://github.com/cowarder/yolov3_voc/blob/master/detect.py): image demo. You can make detection of an image by this file.  

* [train.py](https://github.com/cowarder/yolov3_voc/blob/master/train.py): training model file.  

* [utils.py](https://github.com/cowarder/yolov3_voc/blob/master/utils.py): this file includes necessary utility functions.  

* [voc_label.py](https://github.com/cowarder/yolov3_voc/blob/master/voc_label.py): read voc label(xmin, ymin, xmax, ymax) .xml file, and save it as (x, y, w, h) .txt file.  

* [my_eval.py](https://github.com/cowarder/yolov3_voc/blob/master/my_eval.py): evaluate model performance.  

* [test.py](https://github.com/cowarder/yolov3_voc/blob/master/test.py): my algorithm experiments(this file doesn't relate to YOLOV3 training, ignore it).  

* [valid.py](https://github.com/cowarder/yolov3_voc/blob/master/valid.py): validation of my algorithm(this file doesn't relate to YOLOV3 training, ignore it).  

* [augmentation.py](https://github.com/cowarder/yolov3_voc/blob/master/augmentation.py): data augmentations I tried(this file doesn't relate to YOLOV3 training, ignore it).  

* [yolov1.py](https://github.com/cowarder/yolov3_voc/blob/master/yolov1.py): I tried Yolov1 model(this file doesn't relate to YOLOV3 training, ignore it).  


If you want to train a Yolov3 model on Pascal VOC 2007 dataset, take steps:  
    &emsp; * Prepare your dataset, which including transfer label format and get image name file, refer files in [data](https://github.com/cowarder/yolov3_voc/tree/master/data) folder.  
    &emsp; * After step 1, run train.py, pay attentation to param configuration.  
    &emsp; * Run my_eval.py to evaluate your model performance.  
