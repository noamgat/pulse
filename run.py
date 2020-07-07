from PULSE import PULSE
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import torch
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import argparse
import os

class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png"))
        self.duplicates = duplicates # Number of times to duplicate the image in the dataset to produce multiple HR images

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if(self.duplicates == 1):
            return image,img_path.stem
        else:
            return image,img_path.stem+f"_{(idx % self.duplicates)+1}"

parser = argparse.ArgumentParser(description='PULSE')

#I/O arguments
parser.add_argument('-input_dir', type=str, default='input', help='input data directory')
parser.add_argument('-targets_dir', type=str, default='targets', help='targets data directory')
parser.add_argument('-output_dir', type=str, default='runs', help='output data directory')
parser.add_argument('-output_suffix', type=str, default='0', help='output data directory')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-duplicates', type=int, default=1, help='How many HR images to produce for every image in the input directory')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')

#PULSE arguments
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="100*L2+0.05*GEOCROSS+0*L2_IDENTITY", help='Loss function to use')
parser.add_argument('-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to store and save intermediate HR and LR images during optimization')

kwargs = vars(parser.parse_args())

torch.cuda.set_device(2)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

dataset = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"])
targets_dataset = Images(kwargs["targets_dir"], duplicates=1)
out_path = Path(kwargs["output_dir"])
output_suffix = kwargs["output_suffix"]
out_path.mkdir(parents=True, exist_ok=True)

dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])

model = PULSE(cache_dir=kwargs["cache_dir"])
model = model.cuda()
# model = DataParallel(model)

toPIL = torchvision.transforms.ToPILImage()
# TODO NOAM: Open identity image
target_identity_im = targets_dataset[0][0].cuda()

#from bicubic import BicubicDownsampleTargetSize
#test = BicubicDownsampleTargetSize.downsampling(target_identity_im.unsqueeze(0), (50, 50), mode='area').squeeze(0)
#toPIL(test.cpu().detach().clamp(0, 1)).save('runs/downsample.png')
#toPIL(target_identity_im.cpu().detach().clamp(0, 1)).save('runs/input.png')
#exit(0)

for ref_im, ref_im_name in dataloader:
    ref_im = ref_im.cuda()
    if(kwargs["save_intermediate"]):
        padding = ceil(log10(100))
        for i in range(kwargs["batch_size"]):
            int_path_HR = Path(out_path / ref_im_name[i] / "HR")
            int_path_LR = Path(out_path / ref_im_name[i] / "LR")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        for j,(HR,LR) in enumerate(model(ref_im,target_identity_im,**kwargs)):
            for i in range(kwargs["batch_size"]):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}_{output_suffix}.png")
                toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}_{output_suffix}.png")
    else:
        #out_im = model(ref_im,**kwargs)
        for j,(HR,LR) in enumerate(model(ref_im,target_identity_im,**kwargs)):
            for i in range(kwargs["batch_size"]):
                output_filename = out_path / f"{ref_im_name[i]}_{output_suffix}.png"
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(output_filename)
                print(f"Created {output_filename}")
