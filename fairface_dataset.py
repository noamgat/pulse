import dlib
import kornia
import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from arcface_features_extractor import ArcfaceFeaturesExtractor

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


class FairfacePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_fair_7 = torchvision.models.resnet34(pretrained=True)
        model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
        model_fair_7.load_state_dict(torch.load('fairface/fair_face_models/res34_fair_align_multi_7_20190809.pt'))
        model_fair_7 = model_fair_7.to(device)
        model_fair_7.eval()

        model_fair_4 = torchvision.models.resnet34(pretrained=True)
        model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
        model_fair_4.load_state_dict(torch.load('fairface/fair_face_models/fairface_alldata_4race_20191111.pt'))
        model_fair_4 = model_fair_4.to(device)
        model_fair_4.eval()

        self.model_fair_7 = model_fair_7
        self.model_fair_4 = model_fair_4

    def forward(self, image):
        #print(f'2: {image.shape}')
        image = kornia.resize(image, 224, interpolation='area')
        #print(f'3: {image.shape}')
        image = kornia.normalize(image,
                                 mean=torch.FloatTensor([0.485, 0.456, 0.406]),
                                 std=torch.FloatTensor([0.229, 0.224, 0.225]))
        #print(f'4: {image.shape}')
        outputs = self.model_fair_7(image)
        race_outputs = outputs[:, :7]
        gender_outputs = outputs[:, 7:9]
        age_outputs = outputs[:, 9:18]
        return race_outputs


if __name__ == '__main__':
    dataset = FairfaceDataset()
    print(len(dataset))
    print(dataset[len(dataset)-1])
    print(dataset.races)

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = FairfacePredictor()

    num_correct = 0
    for i, (fn, race_vector) in enumerate(tqdm(dataset)):
        image = dlib.load_rgb_image(fn)
        image = trans(image)
        image = image.view(3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)
        bboxes, landmarks = ArcfaceFeaturesExtractor.get_central_face_attributes(image)
        image = image.squeeze(0)
        image = ArcfaceFeaturesExtractor.align_face(image, landmarks, crop_size=(224, 224))
        #print(f'1: {image.shape}')

        logits = module(image)
        predicted = logits.argmax().detach().cpu().item()
        answer = race_vector.argmax()
        is_correct = predicted == answer
        if is_correct:
            num_correct += 1
        if i > 0 and i % 1000 == 0:
            accuracy = 100.0 * num_correct / (i+1)
            print(f"Intermediate result on {i}: Accuracy = {accuracy:.2f}")
    accuracy = 100 * num_correct / len(dataset)
    print(f"Finished! Accuracy = {accuracy:.2f}")


