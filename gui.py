import sys

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from utils import *
from image import letter_image
from yolo_voc.darknet import Darknet


def detect(cfgfile, weightfile, imgfile, namefile):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    print('Loading weights from {}... Done!'.format(weightfile))

    use_cuda = False
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = letter_image(img, m.width, m.height)

    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4)

    finish = time.time()
    print('{}: Predicted in {:f} seconds.'.format(imgfile, (finish-start)))

    class_names = read_class_names(namefile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)


class MainUI(QWidget):

    def __init__(self):
        super().__init__()
        self.setUI()

    def setUI(self):
        self.resize(1000, 618)
        self.move(100, 100)
        self.setWindowIcon(QIcon('pics/rocket.png'))
        self.setWindowTitle("检测演示")
        self.img_path = None

        inp_btn = QPushButton("选择图片", self)  # self类似于C++ this指针
        inp_btn.setToolTip("点击选择想要检测的图片")
        # btn.clicked.connect(QCoreApplication.instance().quit)
        inp_btn.move(250, 0)
        inp_btn.resize(60, 34)
        inp_btn.clicked.connect(self.openimg)

        pred_btn = QPushButton("进行预测", self)  # self类似于C++ this指针
        pred_btn.setToolTip("对已选择图片检测")
        pred_btn.move(750, 0)
        pred_btn.resize(60, 34)
        pred_btn.clicked.connect(self.show_predict_result)

        self.input_label = QLabel(self)
        self.input_label.resize(0, 0)
        self.input_label.move(160, 160)
        self.input_label.setStyleSheet("QLabel{background:white;}""QLabel{color:rgb(300,300,300,120);"
                                        "font-size:10px;font-weight:bold;font-family:宋体;}")

        self.output_label = QLabel(self)
        self.output_label.resize(0, 0)
        self.output_label.move(540, 160)
        self.output_label.setStyleSheet("QLabel{background:white;}""QLabel{color:rgb(300,300,300,120);"
                                        "font-size:10px;font-weight:bold;font-family:宋体;}")

        self.center()
        self.show()

    """
    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                                     "确定退出？", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    """

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def openimg(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;*.rng;;All Files(*)")
        input_img = QPixmap(imgName)
        self.img_path = imgName

        h = input_img.height()
        w = input_img.width()
        fix_w = 300
        fix_h = 300
        if w*h == 0:
            return
        if fix_w/w < fix_h/h:
            h = (h * fix_w) / w
            w = fix_w
        else:
            w = (w*fix_h)/h
            h = fix_h

        self.input_label.resize(w, h)
        input_img = input_img.scaled(w, h)
        self.input_label.setPixmap(input_img)

        self.output_label.resize(w, h)

    def show_predict_result(self):
        cfgfile = 'data/yolo_v3.cfg'
        weightfile = 'data/model.weights'
        imgfile = self.img_path
        namefile = 'data/voc.names'

        detect(cfgfile, weightfile, imgfile, namefile)
        output_img = QPixmap('predictions.jpg')

        h = output_img.height()
        w = output_img.width()
        fix_w = 300
        fix_h = 300
        if w * h == 0:
            return
        if fix_w / w < fix_h / h:
            h = (h * fix_w) / w
            w = fix_w
        else:
            w = (w * fix_h) / h
            h = fix_h

        output_img = output_img.scaled(w, h)
        self.output_label.setPixmap(output_img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainUI()

    sys.exit(app.exec_())
