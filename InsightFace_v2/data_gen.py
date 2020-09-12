import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from celeba_aligned_copy import build_aligned_celeba, CelebAAdverserialDataset, CelebAPairsDataset
from config import IMG_DIR
from config import pickle_file

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data
        self.transformer = data_transforms['train']

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        label = sample['label']

        filename = os.path.join(IMG_DIR, filename)
        img = Image.open(filename).convert('RGB')
        img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.samples)


class AdverserialFaceDataset(Dataset):
    def __init__(self, split):
        celeba_raw = build_aligned_celeba('../CelebA_Raw', '../CelebA_large', split=split)
        #celeba_orig = CelebA(root='../CelebA_Raw', split='all', download=False, target_type='identity')
        #celeba = celeba_orig

        generated_suffix = 'withidentity'  # can also be 'generated'
        generated = build_aligned_celeba('../CelebA_Raw', f'../CelebA_{generated_suffix}', new_image_suffix='_0', split=split)
        large_matching_generated = build_aligned_celeba('../CelebA_Raw', '../CelebA_large',
                                                        custom_indices=generated.filtered_indices, split=split)
        adverserial_dataset_1 = CelebAAdverserialDataset(generated, large_matching_generated, return_indices=True)
        pairs_dataset = CelebAPairsDataset(celeba_raw, same_ratio=1, num_samples=len(adverserial_dataset_1), return_indices=True)
        self.generated_dataset = adverserial_dataset_1
        self.pairs_dataset = pairs_dataset

        import celeba_eval

        generated_samples_file = 'data/adv_generated.pkl'
        if not os.path.exists(generated_samples_file):
            celeba_eval.process(None, self.generated_dataset, 'data/adv_generated_pairs.txt', generated_samples_file)
        with open(generated_samples_file, 'rb') as file:
            data = pickle.load(file)
            self.generated_samples = data['samples']

        pairs_sample_file = 'data/adv_pairs.pkl'
        if not os.path.exists(pairs_sample_file):
            celeba_eval.process(None, self.pairs_dataset, 'data/adv_pairs_pairs.txt', pairs_sample_file)
        with open(pairs_sample_file, 'rb') as file:
            data = pickle.load(file)
            self.pairs_samples = data['samples']
        self.transformer = data_transforms[split]


    def __len__(self):
        return len(self.generated_dataset) + len(self.pairs_dataset)

    def __getitem__(self, item):
        import celeba_eval
        data_source = self.generated_dataset if (item % 2) == 0 else self.pairs_dataset
        samples = self.generated_samples if (item % 2) == 0 else self.pairs_samples
        data_index = item // 2
        (data_source_1, idx1), (data_source_2, idx2), is_different = data_source[data_index]
        fn1 = data_source_1.filename[idx1]
        fn2 = data_source_2.filename[idx2]
        img1 = celeba_eval.get_image(samples, self.transformer, fn1)
        img2 = celeba_eval.get_image(samples, self.transformer, fn2)
        if img1 is None or img2 is None:
            # Corrupt image or faulty landmark detection
            item = (item + 1) % len(self)
            return self[item]
        return img1, img2, is_different
