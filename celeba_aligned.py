import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
import os
import numpy as np
import torch
import random

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


if __name__ == '__main__':
    large = build_aligned_celeba('CelebA_Raw', 'CelebA_large')
    small = build_aligned_celeba('CelebA_Raw', 'CelebA_small')

    from collections import defaultdict
    identity_dicts = defaultdict(list)
    for idx, identity_idx in enumerate(large.identity):
        identity_dicts[identity_idx.item()].append(idx)

    from bicubic import BicubicDownsampleTargetSize
    from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

    downsample_to_160 = BicubicDownsampleTargetSize(160, True)
    inception_resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    face_features_extractor = torch.nn.Sequential(downsample_to_160, inception_resnet)

    num_trials = 10000
    same_person_deltas = []
    different_person_deltas = []
    identity_indices = list(identity_dicts.keys())
    for _ in range(num_trials):
        identities = np.random.choice(identity_indices, 2, replace=False)
        if min(len(identity_dicts[identities[0]]), len(identity_dicts[identities[1]])) < 2:
            continue
        p1_a, p1_b = np.random.choice(identity_dicts[identities[0]], 2, replace=False)
        p2_a, = np.random.choice(identity_dicts[identities[1]], 1)
        images = [large[idx][0] for idx in (p1_a, p1_b, p2_a)]
        feature_vectors = [face_features_extractor(img.unsqueeze(0)) for img in images]
        delta_same_person = (feature_vectors[1] - feature_vectors[0]).abs().sum().item()
        delta_different = (feature_vectors[2] - feature_vectors[0]).abs().sum().item()
        same_person_deltas.append(delta_same_person)
        different_person_deltas.append(delta_different)
    print(f"Number of experiments: {len(same_person_deltas)}")
    print(f"Average Same Person Delta: {np.mean(same_person_deltas)}. STD: {np.std(same_person_deltas)}")
    print(f"Average Different Person Delta: {np.mean(different_person_deltas)}. STD: {np.std(different_person_deltas)}")
    cutoff_point = (np.mean(different_person_deltas) + np.mean(same_person_deltas)) / 2
    cutoff_accuracy = ((np.array(same_person_deltas) < cutoff_point).sum() + (np.array(different_person_deltas) > cutoff_point).sum()) / (num_trials*2)
    print(f"Cutoff accuracy: {100 * cutoff_accuracy}")
    print("Test complete")
