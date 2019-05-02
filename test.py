from utils import *
from image import letter_image
from darknet import Darknet
from PIL import Image
from tqdm import tqdm
import torch
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

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
    model.load_weights('models_scratch/epoch_40.weights')
    model.eval()
    use_cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if use_cuda else 'cpu'
    model = model.to(device)

    with open("bb_gt_relation.txt", 'w') as f:
        img_ids = open("./data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", ) \
            .read().strip().split()
        for id in tqdm(img_ids):

            img = Image.open('./data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/{}.jpg'
                             .format(id)).convert('RGB')
            label_path = './data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/labels/{}.txt'.format(id)

            img, org_w, org_h = letter_image(img, model.width, model.height), img.width, img.height
            width = img.width
            img = image2tensor(img)
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

            iou_thresh = 0.5
            best_fscore = 0.0
            best_recall = 0.0
            best_precision = 0.0
            besf_thresh = 0.4
            correct_yolo_boxes(boxes, org_w, org_h, model.width, model.height)
            for nms_thresh in np.arange(0.1, 0.91, 0.02):    # np.arange(0.1, 0.8, 0.05):
                nms_thresh = float('{:.2f}'.format(nms_thresh))
                correct = 0.0

                copy_boxes = boxes[:]
                copy_boxes = np.array(nms(copy_boxes, nms_thresh=nms_thresh))

                num_pred = len(copy_boxes)
                if num_pred == 0:
                    continue
                truths = labels.view(-1, 5)
                num_gts = truth_length(truths)
                total = num_gts
                proposals = (copy_boxes[: 4] > 0).sum()

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
                if precision > best_precision:
                    best_fscore = fscore
                    best_precision = precision
                    best_recall = recall
                    besf_thresh = nms_thresh
            # if besf_thresh == 0.1 and num_gts > 5:
            print(label_path, total, correct, besf_thresh)
            # print(str(num_bbs) + " " + str(num_gts) + " " + str(besf_thresh))
            f.write(str(num_bbs) + " " + str(num_gts) + " " + str(best_recall) + " " + str(best_precision) +
                    " " + str(best_fscore) + ' ' + str(besf_thresh) + '\n')


def linear_analysis():

    def pearson(x, y):
        return stats.pearsonr(x, y)

    nmg_range = np.arange(0.4, 0.75, 0.05)

    with open('bb_gt_relation.txt', 'r') as f:
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

    df = pd.DataFrame(columns=['gt_num', 'bb_num'])
    df['gt_num'] = gt_num
    df['bb_num'] = bb_num

    corr = df['gt_num'].corr(df['bb_num'])
    print(corr)
    plt.show()

    # sns.jointplot(df["gt_num"], df["bb_num"], df, annot_kws=dict(stat="r"), kind='kde')
    # g.annotate(pearson, template='r: {val:.2f}\np: {p:.3f}')
    # plt.show()


def get_reg_model():

    with open('bb_gt_relation.txt', 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split() for s in lines]
    lines = [[int(x[0]), int(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in lines]
    lines = [line for line in lines if '{:.2f}'.format(line[5]) != '0.75']
    lines = sorted(lines, key=lambda x: x[1])
    # print(len([x for x in lines if x[0] > 50]))

    bb_num = [x[0] for x in lines]
    gt_num = [x[1] for x in lines]
    recall = [x[2] for x in lines]
    precision = [x[3] for x in lines]
    fscore = [x[4] for x in lines]
    thresh = [x[5] for x in lines]

    df = pd.DataFrame(columns=['gt_num', 'bb_num', 'recall', 'precision', 'fscore', 'thresh'])
    df['bb_num'] = bb_num
    df['gt_num'] = gt_num
    df['recall'] = recall
    df['precision'] = precision
    df['fscore'] = fscore
    df['thresh'] = thresh

    # corr analysis
    # corr = df.corr()
    # print(corr)
    # plt.scatter(list(df['gt_num']), list(df['thresh']))
    # plt.show()

    # linear model analysis between gt_num and thresh
    # reg = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    # gt_num = [[x] for x in gt_num]
    # reg.fit(gt_num, thresh)
    # k = reg.coef_
    # b = reg.intercept_
    # x = np.arange(0, 30, 0.1)
    # y = k * x + b
    # plt.scatter(df['gt_num'], df['thresh'])
    # plt.plot(x, y)
    # plt.show()
    # print(k, b)

    # linear model analysis between bb_num and thresh
    # reg = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    # bb_num = [[x] for x in bb_num]
    # reg.fit(bb_num, thresh)
    # k = reg.coef_
    # b = reg.intercept_
    # x = np.arange(0, 250, 0.1)
    # y = k * x + b
    # plt.scatter(df['bb_num'], df['thresh'])
    # plt.plot(x, y)
    # plt.show()
    # print(k, b)

    # bayes model analysis sbetween bb_num and thresh
    reg = GaussianNB()
    bb_num = [[x] for x in bb_num]
    thresh = [int(x * 100) for x in thresh]
    reg.fit(bb_num, thresh)
    # for i in range(10, 100):
    #     print(reg.predict([[i]])[0]/100)
    return reg

#
# def main():
#     # get_num_file()
#     # linear_analysis()
#     model = get_reg_model()
#
#
# if __name__=='__main__':
#     main()
