from utils import *
from image import letter_image
from yolo_voc.darknet import Darknet


def detect(cfgfile, weightfile, imgfile, namefile):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    use_cuda = False # torch.cuda.is_available()
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = letter_image(img, m.width, m.height)

    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4)

    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = read_class_names(namefile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)


if __name__ == '__main__':
    cfgfile = 'data/yolo_v3.cfg'
    weightfile = 'data/model.weights'
    imgfile = "data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/004408.jpg"
    namefile = 'data/voc.names'
    detect(cfgfile, weightfile, imgfile, namefile)
