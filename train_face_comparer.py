import os

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from celeba_aligned import build_aligned_celeba, CelebAPairsDataset
from face_comparer import FaceComparer
import torch
import torch.nn.functional as F


class FaceComparerTrainer(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_comparer = FaceComparer(True)
        #self.face_comparer.cuda()
        #self.device = self.face_comparer.tail[0].weight.device # TODO : Easiest way?

    def forward(self, x1, x2):
        return self.face_comparer.forward(x1, x2)

    def train_dataloader(self):
        large = build_aligned_celeba('CelebA_Raw', 'CelebA_large')
        pairs_dataset = CelebAPairsDataset(large, same_ratio=0.5)
        return DataLoader(pairs_dataset, batch_size=8)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_nb):
        x1, x2, y = batch
        prediction = self(x1, x2)
        loss = F.binary_cross_entropy_with_logits(prediction.to(torch.double), y.to(torch.double))
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


if __name__ == '__main__':
    torch.cuda.set_device(2)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    trainer = Trainer(gpus=[2])
    net = FaceComparerTrainer()
    trainer.fit(net)
