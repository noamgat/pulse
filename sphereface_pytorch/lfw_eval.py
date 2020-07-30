from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from celeba_aligned_copy import build_aligned_celeba
from mtcnn_pytorch.src.utils import show_bboxes


torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile
from mtcnn_pytorch.src.detector import detect_faces

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere

# mxnet
import mxnet as mx
from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
mxnet_detector = MtcnnDetector(model_folder='mxnet_mtcnn_face_detection/model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=True)

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    # NOAM TEST - CROP INSTEAD OF WARP
    #src_img = src_img[20:-20, 20:-20, :]
    #cropped_img = cv2.resize(src_img, (112, 112))
    #cropped_img = cropped_img[:, 8:-8, :]
    #return cropped_img
    return face_img


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold



parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--lfw', default='../../dataset/face/lfw/lfw.zip', type=str)
parser.add_argument('--model','-m', default='sphere20a.pth', type=str)
args = parser.parse_args()

predicts=[]
net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True

zfile = zipfile.ZipFile(args.lfw)

landmark = {}
with open('data/lfw_landmark.txt') as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [int(k) for k in l[1:]]

with open('data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]

N = 6000

for i in tqdm(range(N)):
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))

    # NOAM: Landmark extraction test
    loaded_landmarks = []
    calculated_landmarks = []
    for j, imname in enumerate([name1, name2]):
        loaded_landmarks.append(landmark[imname])

        raw_img_1 = Image.open(zfile.open(imname))
        #mtcnn_pytorch
        bounding_boxes, landmarks1_detected = detect_faces(raw_img_1)


        cv2_img = cv2.imdecode(np.frombuffer(zfile.read(imname), np.uint8), 1)
        # mxnet
        #mxnet_landmarks = mxnet_detector.detect_face(cv2_img)
        #if mxnet_landmarks:
        #    landmarks1_detected = mxnet_landmarks[1]
        # caffe
        from mtcnn_caffe.demo import complete_detection
        caffe_landmarks = complete_detection(cv2_img, 'mtcnn_caffe/model')
        if caffe_landmarks and len(caffe_landmarks[1]) > 0:
            landmarks1_detected = caffe_landmarks[1]

        txt = [0] * 10
        mtcnn = landmarks1_detected[0]
        txt[0] = mtcnn[0]
        txt[1] = mtcnn[5]
        txt[2] = mtcnn[1]
        txt[3] = mtcnn[6]
        txt[4] = mtcnn[2]
        txt[5] = mtcnn[7]
        txt[6] = mtcnn[3]
        txt[7] = mtcnn[8]
        txt[8] = mtcnn[4]
        txt[9] = mtcnn[9]
        # https://github.com/clcarwin/sphereface_pytorch/issues/4
        landmarks1_detected = list(landmarks1_detected[0])
        #calculated_landmarks.append(landmarks1_detected)
        calculated_landmarks.append(txt)
        if i < 0:
            annotated_image = show_bboxes(raw_img_1, bounding_boxes, [landmarks1_detected])
            annotated_image.save('data/annotation_%d_%d_calc.png' % (i, j))
            original_annotations = show_bboxes(raw_img_1, [], [loaded_landmarks[i]])
            original_annotations.save('data/annotation_%d_%d_orig.png' % (i, j))


            mxnet_image = show_bboxes(raw_img_1, mxnet_landmarks[0], mxnet_landmarks[1])
            mxnet_image.save('data/annotation_%d_%d_mxnet.png' % (i, j))
            caffe_image = show_bboxes(raw_img_1, caffe_landmarks[0], caffe_landmarks[1])
            caffe_image.save('data/annotation_%d_%d_caffe.png' % (i, j))

    used_landmarks = calculated_landmarks  # loaded_landmarks or calculated landmarkss
    img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),used_landmarks[0])
    img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1),used_landmarks[1])

    if i < 5:
        cv2.imwrite('data/input%d_A.jpg' % i, img1)
        cv2.imwrite('data/input%d_B.jpg' % i, img2)

    imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    for i in range(len(imglist)):
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[i] = (imglist[i]-127.5)/128.0

    img = np.vstack(imglist)
    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    output = net(img)
    f = output.data
    f1,f2 = f[0],f[2]
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


accuracy = []
thd = []
folds = KFold(n=N, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(list(map(lambda line:line.strip('\n').split(), predicts)))
for idx, (train, test) in enumerate(folds):
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
