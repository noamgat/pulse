from torch.utils.data import Dataset
import pandas as pd
import os


class FairfaceDataset(Dataset):

    def __init__(self, path='fairface', split='train'):
        self.path = path
        self.labels = pd.read_csv(os.path.join(path, f'fairface_label_{split}.csv'))
        self.race_one_hot = pd.get_dummies(self.labels.race)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = os.path.join(self.path, self.labels.file[item])
        attr_vector = self.race_one_hot.values[item]
        return image_path, attr_vector

    @property
    def races(self):
        return list(self.race_one_hot.columns)


if __name__ == '__main__':
    dataset = FairfaceDataset()
    print(len(dataset))
    print(dataset[len(dataset)-1])
    print(dataset.races)


