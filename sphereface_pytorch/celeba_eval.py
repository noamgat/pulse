from __future__ import print_function

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from InsightFace_v2.utils import get_central_face_attributes_img
from celeba_aligned_copy import build_aligned_celeba, CelebAPairsDataset
from mtcnn_pytorch.src.detector import detect_faces

torch.backends.cudnn.bencmark = True

import cv2
import argparse
import numpy as np

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
    cropped_img = cv2.resize(src_img, (112, 112))
    cropped_img = cropped_img[:, 8:-8, :]
    return cropped_img
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
parser.add_argument('--celeba_orig', default='../CelebA_Raw', type=str)
parser.add_argument('--celeba_new', default='../CelebA_large', type=str)
parser.add_argument('--model','-m', default='sphere20a.pth', type=str)
args = parser.parse_args()

predicts=[]
net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True

# zfile = zipfile.ZipFile(args.lfw)

# landmark = {}
# with open('data/lfw_landmark.txt') as f:
#     landmark_lines = f.readlines()
# for line in landmark_lines:
#     l = line.replace('\n','').split('\t')
#     landmark[l[0]] = [int(k) for k in l[1:]]
#
# with open('data/pairs.txt') as f:
#     pairs_lines = f.readlines()[1:]

N = 6000

celeba = build_aligned_celeba(args.celeba_orig, args.celeba_new)
celeba_dataset = CelebAPairsDataset(celeba, num_samples=N)

def detect_landmarks(pil_im):
    # mtcnn_pytorch
    bounding_boxes, landmarks1_detected = detect_faces(pil_im)

    cv2_img = np.array(pil_im)

    central_boxes, central_landmarks = get_central_face_attributes_img(cv2_img)
    if bounding_boxes is not None:
        landmarks1_detected = central_landmarks
    # mxnet
    #mxnet_landmarks = mxnet_detector.detect_face(cv2_img)
    #if mxnet_landmarks:
    #    landmarks1_detected = mxnet_landmarks[1]
    # caffe
    #
    #caffe_landmarks = complete_detection(cv2_img, 'mtcnn_caffe/model')
    #if caffe_landmarks and len(caffe_landmarks[1]) > 0:
    #    landmarks1_detected = caffe_landmarks[1]

    txt = [0] * 10
    if landmarks1_detected is None or len(landmarks1_detected) == 0:
        return None
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
    return txt

for i in tqdm(range(N)):
    # p = pairs_lines[i].replace('\n','').split('\t')
    #
    # if 3==len(p):
    #     sameflag = 1
    #     name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
    #     name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    # if 4==len(p):
    #     sameflag = 0
    #     name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
    #     name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
    #
    # img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),landmark[name1])
    # img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1),landmark[name2])
    toPIL = torchvision.transforms.ToPILImage()
    img1, img2, is_different = celeba_dataset[i]
    # img1 = img1[:, :, 8:-8]
    # img2 = img2[:, :, 8:-8]
    pil_1 = toPIL(img1)
    pil_2 = toPIL(img2)
    sameflag = 1 - is_different
    if i < 5:

        # im1, im2, is_same = adverserial_dataset_1[99]
        # toPIL(im1).save('im1.png')
        pil_1.save('data/input_celeba_%d_A.jpg' % i)
        pil_2.save('data/input_celeba_%d_B.jpg' % i)

    #imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    #for i in range(len(imglist)):
    #    imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
    #    imglist[i] = (imglist[i]-127.5)/128.0
    landmarks_1 = detect_landmarks(pil_1)
    landmarks_2 = detect_landmarks(pil_2)
    if not (landmarks_1 and landmarks_2):
        print("WAH")
        continue
    img1 = alignment(np.array(pil_1), landmarks_1)
    img2 = alignment(np.array(pil_2), landmarks_2)
    if i < 5:
        Image.fromarray(img1).save('data/input_celeba_%d_A_aligned.jpg' % i)
        Image.fromarray(img2).save('data/input_celeba_%d_B_aligned.jpg' % i)
    #img = np.vstack(imglist)
    toTensor = torchvision.transforms.ToTensor()
    img = torch.stack([toTensor(img1), toTensor(img2)])
    #img = img[:, :, 8:-8, :]
    img = (img - 0.5) * 2
    #img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    img = img.cuda()
    output = net(img)
    f = output.data
    f1, f2 = f[0],f[1]
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(f'{i}_A',f'{i}_B',cosdistance,sameflag))


accuracy = []
thd = []
folds = KFold(n=len(predicts), n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(list(map(lambda line:line.strip('\n').split(), predicts)))
for idx, (train, test) in enumerate(folds):
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('CELEBAACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
