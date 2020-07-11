import os
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from celeba_aligned import build_aligned_celeba, CelebAPairsDataset
from face_comparer import FaceComparer
import torch
import torch.nn.functional as F

import pl_transfer_learning_helpers

class FaceComparerTrainer(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_comparer = FaceComparer(True)
        pl_transfer_learning_helpers.freeze(self.face_comparer.face_features_extractor, train_bn=False)
        #self.face_comparer.cuda()
        #self.device = self.face_comparer.tail[0].weight.device # TODO : Easiest way?

    def forward(self, x1, x2):
        return self.face_comparer.forward(x1, x2)

    def get_dataloader(self, split='train', same_ratio=0.5, batch_size=16):
        large = build_aligned_celeba('CelebA_Raw', 'CelebA_large', split=split)
        pairs_dataset = CelebAPairsDataset(large, same_ratio=same_ratio, num_samples=10000)
        return DataLoader(pairs_dataset, batch_size=batch_size, num_workers=2)

    @pl.data_loader
    def train_dataloader(self):
        return self.get_dataloader()

    @pl.data_loader
    def val_dataloader(self):
        return self.get_dataloader(split='valid')

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_nb):
        x1, x2, y = batch
        y = y.unsqueeze(1)
        prediction = self(x1, x2)
        num_correct = int(((prediction.sign() / 2) + 0.5 == y).to(float).sum().item())
        loss = F.binary_cross_entropy_with_logits(prediction.to(torch.double), y.to(torch.double))
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs, 'num_correct': num_correct}

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y = y.unsqueeze(1)

        # implement your own
        prediction = self(x1, x2)
        loss = F.binary_cross_entropy_with_logits(prediction.to(torch.double), y.to(torch.double))
        num_correct = int(((prediction.sign() / 2) + 0.5 == y).to(float).sum().item()) / (len(y) * 1.0)


        # all optional...
        # return whatever you need for the collation function test_end
        output = OrderedDict({
            'val_loss': loss,
            'val_acc': torch.tensor(num_correct),  # everything must be a tensor
        })

        # return an optional dict
        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        results = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        print(f"Epoch {self.current_epoch} Validation: Acc = {val_acc_mean.item()}, Loss = {val_loss_mean.item()}")
        return {'progress_bar': results, 'log': results, 'val_loss': results['val_loss']}


if __name__ == '__main__':
    torch.cuda.set_device(2)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    trainer = Trainer(gpus=[2])
    net = FaceComparerTrainer()
    trainer.fit(net)
