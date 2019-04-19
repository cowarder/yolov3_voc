from utils import *
from yolo_voc.darknet import Darknet
import cv2
import sys


def demo(cfgfile, weightfile, namefile):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    class_names = read_class_names(namefile)

    use_cuda = False
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    while True:
        res, img = cap.read()

        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            print('------')
            draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
            cv2.imshow(cfgfile, draw_img)
            if len(bboxes) != 0:
                cv2.waitKey(1)
        else:
            print("Unable to read image")
            exit(-1)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        namefile = sys.argv[3]
        demo(cfgfile, weightfile, namefile)
        # %run camera.py data/yolo_v3.cfg data/model.weights data/voc.names
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')
