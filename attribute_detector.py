import argparse
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA

parser = argparse.ArgumentParser(description='Attribute Detector')
parser.add_argument('-gpu_id', default='2', type=str, help='Which gpu to use. Can also use multigpu format')
parser.add_argument('-face_comparer_config', default='configs/arcface_adv.yml', type=str, help='YML file of face comparer')
parser.add_argument('-batch_size', type=int, default=16, help='Batch size to use during optimization')
parser.add_argument('-ckpt', type=str, default=None, help='Checkpoint to start training from')
kwargs = vars(parser.parse_args())

#FEATURE_INDEX = 0

def build_mlp(input_dim, hidden_dims, output_dim):
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(1, len(dims)):
        layers.append(torch.nn.Linear(dims[i-1], dims[i]))
        if i < len(dims)-1:
            layers.append(torch.nn.BatchNorm1d(num_features=dims[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

class FeaturesDataset(Dataset):
    def __init__(self, feature_dict, split):
        celeb_a_train = CelebA(root='CelebA_Raw', split=split)
        self.feature_attrib_pairs = []
        for i, fn in enumerate(celeb_a_train.filename):
            fn = os.path.splitext(fn)[0]
            if fn in feature_dict:
                feature_vector = feature_dict[fn]
                attrib_vector = celeb_a_train.attr[i]
                #attrib_vector = attrib_vector[FEATURE_INDEX:FEATURE_INDEX+1]
                self.feature_attrib_pairs.append((feature_vector, attrib_vector))

    def __len__(self):
        return len(self.feature_attrib_pairs)

    def __getitem__(self, item):
        feature, attrib = self.feature_attrib_pairs[item]
        feature_vector = torch.FloatTensor(feature)
        attrib_vector = attrib.type(torch.float)
        return feature_vector, attrib_vector

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.l1 = torch.nn.Linear(512, 40)
        self.l1 = build_mlp(512, [1024, 512], 40)

    def forward(self, x):
        logits = self.l1(x)
        return torch.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        num_correct = int((y_hat.round() == y).to(float).mean().item()*100)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs, 'num_correct': num_correct}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        num_correct = int((y_hat.round() == y).to(float).mean().item() * 100)
        accuracy_per_attr = (y_hat.round() == y).float().mean(dim=0).detach().cpu()
        metrics = {'val_acc': num_correct, 'val_loss': loss, 'val_acc_per_attr': accuracy_per_attr}
        return metrics

    def validation_epoch_end(self, outputs):
        accs = [float(output['val_acc']) for output in outputs]
        val_loss = torch.stack([output['val_loss'] for output in outputs]).mean ()
        acc = sum(accs) / len(accs)
        print(f"Epoch Accuracy: {acc:.2f}")
        attr_accs = [output['val_acc_per_attr'] for output in outputs]
        attr_acc = (torch.stack(attr_accs).mean(dim=0) * 100).round()
        print(f"Per attribute accuracy:")
        print(attr_acc.tolist())
        return {'Training/_accuracy': acc, 'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


gpu_id = kwargs['gpu_id']
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
#torch.cuda.set_device(gpu_id)


features_file = kwargs['face_comparer_config'] + '.features.json'
if not os.path.exists(features_file):
    raise Exception(f"Features json f{features_file} does not exist")
feature_dict = json.load(open(features_file, "r"))

dataset_train = FeaturesDataset(feature_dict, 'train')
print(f"Loaded dataset with {len(dataset_train)} feature vectors")
dataset_valid = FeaturesDataset(feature_dict, 'valid')
train_loader = DataLoader(dataset_train, batch_size=kwargs['batch_size'], shuffle=True)#, num_workers=2)
val_loader = DataLoader(dataset_valid, batch_size=kwargs['batch_size'])#, num_workers=2)
config_dir, config_file = os.path.split(kwargs['face_comparer_config'])
attribute_model_file = os.path.splitext(config_file)[0]
checkpoint_callback = ModelCheckpoint(config_dir, save_weights_only=True, prefix=attribute_model_file)

trainer = pl.Trainer(gpus=[torch.cuda.current_device()],
                     checkpoint_callback=checkpoint_callback)
if kwargs['ckpt']:
    model = LitModel.load_from_checkpoint(kwargs['ckpt'], map_location='cuda:0')
else:
    model = LitModel()

trainer.fit(model, train_loader, val_dataloaders=val_loader)
