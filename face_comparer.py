from bicubic import BicubicDownsampleTargetSize
from stylegan import G_synthesis,G_mapping
from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import torch
from loss import LossBuilder
from functools import partial
from drive import open_url
from facenet_pytorch import InceptionResnetV1


def build_mlp(input_dim, hidden_dims, output_dim):
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(1, len(dims)):
        layers.append(torch.nn.Linear(dims[i-1], dims[i]))
        if i < len(dims)-1:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class FaceComparer(torch.nn.Module):
    def __init__(self, load_pretrained=True):
        super(FaceComparer, self).__init__()
        downsample_to_160 = BicubicDownsampleTargetSize(160, True)
        pretrained_name = 'vggface2' if load_pretrained else None
        inception_resnet = InceptionResnetV1(pretrained=pretrained_name, classify=False)
        self.face_features_extractor = torch.nn.Sequential(downsample_to_160, inception_resnet)
        self.tail = build_mlp(512, [], 1)

    def forward(self, x_1, x_2):
        features_1 = self.face_features_extractor(x_1)
        features_2 = self.face_features_extractor(x_2)
        features_diff = features_1 - features_2
        mlp_output = self.tail(features_diff)
        decision = torch.sigmoid(mlp_output)
        return decision
