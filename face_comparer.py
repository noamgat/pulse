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
    def __init__(self, load_pretrained=True, hidden_dims=[], initial_bias=None):
        super(FaceComparer, self).__init__()
        downsample_to_160 = BicubicDownsampleTargetSize(160, True)
        pretrained_name = 'vggface2' if load_pretrained else None
        inception_resnet = InceptionResnetV1(pretrained=pretrained_name, classify=False).eval()
        self.face_features_extractor = torch.nn.Sequential(downsample_to_160, inception_resnet)
        self.tail = build_mlp(512, hidden_dims, 1)
        last_fc = self.tail[0]
        if initial_bias is not None:
            last_fc.weight.data = torch.ones_like(last_fc.weight.data)
            last_fc.bias.data = torch.ones_like(last_fc.bias.data) * initial_bias
        print("Done")

    def forward(self, x_1, x_2):
        features_1 = self.face_features_extractor(x_1)
        features_2 = self.face_features_extractor(x_2)
        features_diff = features_1 - features_2
        # TODO: ABS? Multiply by sign of first element? Square?
        features_diff = abs(features_diff)
        mlp_output = self.tail(features_diff)
        #mlp_output = mlp_output.squeeze(1)
        # decision = torch.sigmoid(mlp_output) #Using BCE loss, that will sigmoid
        decision = mlp_output
        threshold_decision = features_diff.sum(dim=1)
        threshold_decision = threshold_decision - 21
        threshold_decision = threshold_decision.unsqueeze(1)
        #return threshold_decision
        return decision
