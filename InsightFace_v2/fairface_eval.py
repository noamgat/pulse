import torchvision
from PIL import Image
from tqdm import tqdm

from data_gen import FairfaceImageDataset
from fairface_dataset import FairfaceDataset


def fairface_test(feature_extractor, tail_fc):
    feature_extractor.eval()
    tail_fc.eval()
    test_dataset = FairfaceImageDataset(split='val')
    num_correct = 0
    num_images = 0
    try:
        for im, attr_vector in tqdm(test_dataset):
            im = im.unsqueeze(0)
            feature_vector = feature_extractor(im.cuda())
            race_vector = tail_fc(feature_vector)
            selected_race = race_vector[0].argmax()
            correct_race = attr_vector.argmax()
            is_correct = selected_race == correct_race
            if is_correct:
                num_correct += 1
            num_images += 1
    except Exception as e:
        pass

    accuracy = num_correct / num_images
    print(f"TEST FAIRFACE accuracy : {num_correct} / {num_images} ({accuracy*100:.2f}%)")
    return accuracy, 70
