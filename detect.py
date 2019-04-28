from utils import *
from image import letter_image
from darknet import Darknet


def detect(cfgfile, weightfile, imgfile, namefile):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    use_cuda = False  # if torch.cuda.is_available() else False
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = letter_image(img, m.width, m.height)

    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda=use_cuda)

    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = read_class_names(namefile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)


if __name__ == '__main__':
    cfgfile = 'data/yolo_v3.cfg'
    weightfile = 'models/epoch_4.weights'
    # weightfile = 'data/model.weights'
    imgfile = "data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000369.jpg"
    namefile = 'data/voc.names'
    detect(cfgfile, weightfile, imgfile, namefile)
