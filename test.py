
from utils import *
from image import letter_image
from yolo_voc.darknet import Darknet
from PIL import Image
from tqdm import tqdm
import torch
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

eps = 1e-5


def get_num_file():
    global eps

    def truth_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
        return 50

    global model
    model = Darknet('data/yolo_v3.cfg')
    model.load_weights('data/model.weights')
    model.eval()
    use_cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if use_cuda else 'cpu'
    model = model.to(device)

    with open("result.txt", 'w') as f:
        img_ids = open("./data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", ) \
            .read().strip().split()
        for id in tqdm(img_ids):

            total = 0.0
            proposals = 0.0
            correct = 0.0

            img = Image.open('./data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/{}.jpg'
                             .format(id)).convert('RGB')
            label_path = './data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/labels/{}.txt'.format(id)

            img, org_w, org_h = letter_image(img, model.width, model.height), img.width, img.height
            width = img.width
            img = image2tensor(img)
            if use_cuda:
                img = img.to(device)

            out_boxes = model(img)
            shape = (model.width, model.height)
            boxes = get_all_boxes(out_boxes, shape, 0.5, model.num_classes, device=device)[0]

            num_bbs = len(boxes)
            with open(label_path, 'r') as imgf:
                gt_num = len(imgf.readlines())

            labels = torch.zeros(50 * 5)
            try:
                tmp = torch.from_numpy(read_truths_args(label_path, 8.0 / width).astype('float32'))
            except Exception as e:
                print(e.value)
                tmp = torch.zersos(1, 5)
            tmp = tmp.view(-1)
            tsz = tmp.numel()  # element number
            
            if tsz > 50 * 5:
                labels = tmp[0:50 * 5]
            elif tsz > 0:
                labels[0:tsz] = tmp

            nms_thresh = 0.4
            iou_thresh = 0.5
            best_fscore = -1
            best_recall = -1
            best_precision = -1
            besf_thresh = 0.4
            correct_yolo_boxes(boxes, org_w, org_h, model.width, model.height)
            for nms_thresh in np.arange(0.3, 0.8, 0.05):
                copy_boxes = boxes[:]
                copy_boxes = np.array(nms(copy_boxes, nms_thresh=nms_thresh))

                num_pred = len(copy_boxes)
                if num_pred == 0:
                    continue
                truths = labels.view(-1, 5)
                num_gts = truth_length(truths)
                total += num_gts
                proposals += (copy_boxes[: 4] > 0).sum()

                for k in range(num_gts):
                    gt_box = torch.FloatTensor([truths[k][1], truths[k][2],
                                                truths[k][3], truths[k][4], 1.0, 1.0, truths[k][0]])
                    gt_box = gt_box.repeat(num_pred, 1).t()
                    pred_box = torch.FloatTensor(copy_boxes).t()
                    best_iou, best_j = torch.max(cal_ious(gt_box, pred_box), 0)
                    if best_iou > iou_thresh and pred_box[6][best_j] == gt_box[6][0]:
                        correct += 1

                precision = 1.0 * correct / (proposals + eps)
                recall = 1.0 * correct / (total + eps)
                fscore = 2.0 * precision * recall / (precision + recall + eps)
                # print('correct:{}'.format(correct))
                # print('proposals:{}'.format(proposals))
                # print('total:{}'.format(total))
                if fscore > best_fscore:
                    best_fscore = fscore
                    best_precision = precision
                    best_recall = recall
                    besf_thresh = nms_thresh

            # print(str(num_bbs) + " " + str(num_gts) + " " + str(besf_thresh))
            f.write(str(num_bbs) + " " + str(num_gts) + " " +
                    str(best_recall) + " " + str(best_precision) + " " + str(besf_thresh) + '\n')


def plot_num():

    def pearson(x, y):
        return stats.pearsonr(x, y)

    nmg_range = np.arange(0.4, 0.75, 0.05)

    with open('result.txt', 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split() for s in lines]
    lines = [[int(x[0]), int(x[1])] for x in lines]
    lines = sorted(lines, key=lambda x: x[1])

    bb_num = [x[0] for x in lines]
    gt_num = [x[1] for x in lines]
    plt.figure(figsize=(12.8, 9.6))
    plt.suptitle('相关性分析')

    plt.subplot(3, 2, 1)
    plt.title('散点图')
    plt.scatter(gt_num, bb_num)

    plt.subplot(3, 2, 2)
    plt.title('折线图')
    plt.plot(gt_num, bb_num)

    plt.subplot(3, 2, 3)
    plt.title('ground truth')
    plt.hist(gt_num, color="#FF0000")

    plt.subplot(3, 2, 4)
    plt.title('bounding box')
    plt.hist(bb_num, color="#C1F320")

    plt.subplot(3, 2, 5)
    plt.title('ground truth')
    plt.boxplot(gt_num)

    plt.subplot(3, 2, 6)
    plt.title('bounding box')
    plt.boxplot(bb_num)

    df = pd.DataFrame(columns=['gt', 'bb'])
    df['gt'] = gt_num
    df['bb'] = bb_num

    corr = df['gt'].corr(df['bb'])
    print(corr)
    plt.show()

    # sns.jointplot(df["gt"], df["bb"], df, annot_kws=dict(stat="r"), kind='kde')
    # g.annotate(pearson, template='r: {val:.2f}\np: {p:.3f}')
    # plt.show()


def main():
    get_num_file()
    # plot_num()


if __name__=='__main__':
    main()
