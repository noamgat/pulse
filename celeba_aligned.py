import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
import os
import numpy as np
import torch
import random

from tqdm import tqdm


def build_aligned_celeba(orig_celeba_folder, new_celeba_folder):
    celeb_a = CelebA(root=orig_celeba_folder,
                     split='all',
                     download=False,
                     target_type='identity',
                     transform=torchvision.transforms.ToTensor())
    celeb_a.root = new_celeba_folder
    celeb_a.filename = [f"{os.path.splitext(fn)[0]}_0.png" for fn in celeb_a.filename]
    img_folder = os.path.join(new_celeba_folder, celeb_a.base_folder, "img_align_celeba")
    existing_indices = [os.path.exists(os.path.join(img_folder, fn)) for fn in celeb_a.filename]
    print(f"{sum(existing_indices)} / {len(celeb_a.filename)} images exist in {new_celeba_folder}")

    for list_attr in ['filename', 'identity', 'bbox', 'landmarks_align', 'attr']:
        attr_val = getattr(celeb_a, list_attr)
        filtered_list = np.array(attr_val)[existing_indices]
        if isinstance(attr_val, torch.Tensor):
            filtered_list = torch.Tensor(filtered_list).to(dtype=attr_val.dtype)
        else:
            filtered_list = list(filtered_list)
        setattr(celeb_a, list_attr, filtered_list)
    return celeb_a


class CelebAPairsDataset(Dataset):
    def __init__(self, celeb_a: CelebA, same_ratio=0.5):
        super(CelebAPairsDataset, self).__init__()
        self.celeb_a = celeb_a
        from collections import defaultdict
        identity_dicts = defaultdict(list)
        for idx, identity_idx in enumerate(large.identity):
            identity_dicts[identity_idx.item()].append(idx)
        self.identity_dicts = identity_dicts
        self.identity_indices = list(self.identity_dicts.keys())
        self.same_ratio = same_ratio

    def __len__(self):
        return len(self.celeb_a) * (len(self.celeb_a) + 1) / 2

    def __getitem__(self, item):
        is_same = int(item * self.same_ratio) < int((item + 1) * self.same_ratio)
        if is_same:
            identity = 0
            while len(self.identity_dicts[identity]) < 2:
                identity, = np.random.choice(self.identity_indices, 1)
            idx1, idx2 = np.random.choice(self.identity_dicts[identity], 2, replace=False)
        else:
            identities = np.random.choice(self.identity_indices, 2, replace=False)
            idx1, = np.random.choice(self.identity_dicts[identities[0]], 1)
            idx2, = np.random.choice(self.identity_dicts[identities[1]], 1)
        return self.celeb_a[idx1][0], self.celeb_a[idx2][0], 1 - int(is_same)

if __name__ == '__main__':
    large = build_aligned_celeba('CelebA_Raw', 'CelebA_large')
    small = build_aligned_celeba('CelebA_Raw', 'CelebA_small')

    pairs_dataset = CelebAPairsDataset(large, same_ratio=0.5)

    from bicubic import BicubicDownsampleTargetSize
    from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

    downsample_to_160 = BicubicDownsampleTargetSize(160, True)
    inception_resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    face_features_extractor = torch.nn.Sequential(downsample_to_160, inception_resnet)

    num_trials = 1000
    same_person_deltas = []
    different_person_deltas = []
    for i in tqdm(range(num_trials)):
        p1, p2, is_different = pairs_dataset[i]
        images = [p1, p2]
        feature_vectors = [face_features_extractor(img.unsqueeze(0)) for img in images]
        delta_feature = (feature_vectors[1] - feature_vectors[0]).abs().sum().item()
        if is_different:
            different_person_deltas.append(delta_feature)
        else:
            same_person_deltas.append(delta_feature)
    print(f"Number of experiments: {num_trials}")
    m1 = np.mean(same_person_deltas)
    std1 = np.std(same_person_deltas)
    m2 = np.mean(different_person_deltas)
    std2 = np.std(different_person_deltas)
    print(f"Average Same Person Delta: {m1}. STD: {std1}")
    print(f"Average Different Person Delta: {m2}. STD: {std2}")
    cutoff_point = m1 + (m2 - m1) * (std1 / (std1 + std2))
    cutoff_accuracy = ((np.array(same_person_deltas) < cutoff_point).sum() + (np.array(different_person_deltas) > cutoff_point).sum()) / num_trials
    print(f"Cutoff training accuracy: {100 * cutoff_accuracy}")
    print("Test complete")
