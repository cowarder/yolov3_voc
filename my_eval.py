import os
import sys
import _pickle as cPickle
import numpy as np
sys.path.append("stats")
from utils import read_class_names
import xml.etree.ElementTree as ET


def get_image_xml_name(imagefile):
    return imagefile.replace('images', 'Annotations').replace('JPEGImages', 'Annotations')\
        .replace('.jpg', '.xml').replace('.png', '.xml')


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects


def compute_ap(rec, prec, use_07_metric=False):
    """ ap = compute_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def my_eval(detpath, imagesetfile, classname, cachedir,
            ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = my_eval(detpath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            imagekey = os.path.basename(imagename).split('.')[0]    
            recs[imagekey] = parse_rec(get_image_xml_name(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        imagekey = os.path.basename(imagename).split('.')[0]
        try:
            R = [obj for obj in recs[imagekey] if obj['name'] == classname]
        except:
            print("%s %s" % (imagename, imagekey))
            exit(0)

        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagekey] = {'bbox': bbox,
                                'difficult': difficult,
                                'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    if len(sorted_ind) > 0:
        BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = compute_ap(rec, prec, use_07_metric)

    # print('class: {:<10s} \t num occurrence: {:4d}'.format(classname, npos))

    return rec, prec, ap, npos


def _do_python_eval(res_prefix, imagesetfile, classesfile, output_dir='output'):
    
    filename = res_prefix + '{:s}.txt'

    cachedir = os.path.join(output_dir, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = False
    # print ('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    _classes = read_class_names(classesfile)
    total = 0
    for i, cls in enumerate(_classes):
        if cls == '__background__':
            continue
      
        rec, prec, ap, noccur = my_eval(
            filename, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        total += noccur
        print('AP for {:<10s} = {:.4f} with {:4d} views'.format(cls, ap, noccur))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    print('Mean AP = {:.4f} with total {:4d} views'.format(np.mean(aps), total))

    print(' '*10, 'Results:')
    print('-'*30)
    for i, ap in enumerate(aps):
        print('{:<10s}\t{:.3f}'.format(_classes[i], ap))
    print('='*30)
    print('{:^10s}\t{:.3f}'.format('Average', np.mean(aps)))
    return [res_prefix, np.mean(aps)]


if __name__ == '__main__':
    # res_prefixc = '/data/hongji/darknet/results/comp4_det_test_'
    # res_prefix = 'results/comp4_det_test_'
    # test_file = 'data/sketch_test.txt'
    # class_names = 'data/sketch.names'

    res_prefix = 'result/epoch_36/'
    test_file = 'data/2007_test.txt'
    class_names = 'data/voc.names'
    # _do_python_eval(res_prefix, test_file, class_names, output_dir='output')

    # multi model evals
    res_prefix = 'result/'
    result = []
    for file_name in os.listdir('./result'):
        prefix = res_prefix + file_name + '/'
        result.append(_do_python_eval(prefix, test_file, class_names, output_dir='output'))
    print(result)
    with open('validation.txt', 'w') as f:
        for item in result:
            f.write((item[0].split('/')[1]).split('_')[1])
            f.write('\t')
            f.write(str(item[1]))
            f.write('\n')

