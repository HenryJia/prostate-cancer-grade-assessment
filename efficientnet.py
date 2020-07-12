import time
import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import cv2
import PIL.Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

from albumentations import Compose, HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue, RandomBrightness, RandomContrast, RandomGamma

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger

from dataset import PandaDataset
from efficientnet_pytorch import model as enet
from loader import AsynchronousLoader
from scheduler import GradualWarmupScheduler

import matplotlib.pyplot as plt
plt.ion()

from tqdm import tqdm

class EfficientNetV2(LightningModule):
    def __init__(self, enet_type, pretrained_model, out_dim, precision, epochs, batch_size, num_workers, q_size, lr, num_patches, patch_size, level, **kwargs):
        super(EfficientNetV2, self).__init__()
        self.enet_type = enet_type
        self.pretrained_model = pretrained_model
        self.out_dim = out_dim
        self.precision = precision

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.q_size = q_size

        self.lr = lr

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.level = level

        self.save_hyperparameters()

        self.enet = enet.EfficientNet.from_name(enet_type)
        self.enet.load_state_dict(torch.load(pretrained_model))

        self.fc_out = nn.Linear(self.enet._fc.in_features, out_dim)
        self.mask_out = nn.Conv2d(self.enet._fc.in_features, 2, 1)

    def forward(self, x, mask=False):
        batch_size, num_patches, channels, height, width = x.shape

        # Apply a separate identical enet on every separate patch
        x = x.view(batch_size * num_patches, channels, height, width)
        features = self.enet.extract_features(x)

        x = features.view(batch_size, num_patches, features.shape[1], -1) # batch_size, num_patches, channels, spatial dimension
        x = torch.max(torch.mean(x, dim=3), dim=1)[0]

        x = self.fc_out(x)

        if mask:
            return x, self.mask_out(features).view(batch_size, num_patches, 2, features.shape[-2], features.shape[-1])
        else:
            return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return [optimizer], [scheduler]

    def prepare_data(self):
        root_path = f'/home/nvme/Kaggle/prostate-cancer-grade-assessment/'

        df = pd.read_csv(root_path + 'train.csv')

        mask_present = [] # Only about 100 images in the dataset have no mask so just ignore them for training
        for idx in df['image_id']:
            mask_present += [os.path.isfile(os.path.join(root_path, 'train_label_masks', idx + '_mask.tiff'))]
        df = df[mask_present]

        train_df, validation_df = train_test_split(df, test_size=0.1)

        transforms = Compose([Transpose(p=0.5),
                              VerticalFlip(p=0.5),
                              HorizontalFlip(p=0.5),
                              RandomBrightness(p=0.5, limit=0.2),
                              RandomContrast(p=0.5, limit=0.2)
                              ])
        self.train_set = PandaDataset(root_path, train_df, level=self.level, patch_size=self.patch_size, num_patches=self.num_patches, use_mask=True, transforms=transforms)
        self.validation_set = PandaDataset(root_path, validation_df, level=self.level, patch_size=self.patch_size, num_patches=self.num_patches, use_mask=False)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=self.num_workers)
        return AsynchronousLoader(dataloader, device=torch.device('cuda', 0), q_size=self.q_size, dtype=torch.float16 if args.precision == 16 else torch.float32)

    def val_dataloader(self):
        dataloader = DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False, pin_memory=False, num_workers=self.num_workers)
        return AsynchronousLoader(dataloader, device=torch.device('cuda', 0), q_size=self.q_size, dtype=torch.float16 if args.precision == 16 else torch.float32)

    def training_step(self, batch, batch_idx):
        x, (mask, y) = batch
        batch_size, num_patches, channels, height, width = x.shape

        if self.precision == 16:
            x = x.half()
        else:
            x = x.float()

        out, out_mask = self(x / 255.0, mask=True)

        mask = F.adaptive_max_pool2d(mask.view(batch_size * num_patches, 2, height, width), out_mask.shape[-2:])
        mask = mask.view(batch_size, num_patches, 2, mask.shape[-2], mask.shape[-1])

        loss = F.binary_cross_entropy_with_logits(out, y) + F.binary_cross_entropy_with_logits(out_mask, mask)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.precision == 16:
            x = x.half()
        else:
            x = x.float()

        out = self(x / 255.0)
        loss = F.binary_cross_entropy_with_logits(out, y)

        return {'val_loss': loss, 'out': out.detach(), 'y': y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs], dim=0).mean()
        out = torch.sigmoid(torch.cat([x['out'] for x in outputs], dim=0)).cpu().numpy()
        y = torch.cat([x['y'] for x in outputs], dim=0).cpu().numpy()

        kappa = torch.tensor(cohen_kappa_score(np.sum(out, axis=-1).round(), y.sum(axis=-1), weights='quadratic'))

        logs = {'val_loss': avg_loss, 'kappa': kappa}
        return {'val_loss': avg_loss, 'kappa': kappa, 'log': logs}

argument_parser = ArgumentParser(add_help=False)
argument_parser.add_argument('--enet_type', type=str, default='efficientnet-b0', help='Type of efficientnet to use')
argument_parser.add_argument('--pretrained_model', type=str, default='EfficientNet-PyTorch/efficientnet-b0-08094119.pth', help='location of pretraiend efficientnet')
argument_parser.add_argument('--epochs', type=int, default=10, help='training epochs')
argument_parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
argument_parser.add_argument('--num_workers', type=int, default=16, help='number of workers for dataloaders')
argument_parser.add_argument('--q_size', type=int, default=5, help='queue size for asynchronous loading')
argument_parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
argument_parser.add_argument('--num_patches', type=int, default=20, help='number of patches to take')
argument_parser.add_argument('--patch_size', type=int, default=256, help='size of patches')
argument_parser.add_argument('--level', type=int, default=1, help='resolution level of images')
argument_parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
argument_parser.add_argument('--precision', type=int, default=32, help='which gpu to use')
args = argument_parser.parse_args()

logger = TensorBoardLogger("tb_logs", name="efficientnet")

model_name = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii') + '-' + args.enet_type
print(model_name)

checkpoint_callback = ModelCheckpoint(filepath='./efficientnet-ckpt/'+model_name+'-{epoch:02d}-{kappa:.2f}', save_top_k=1, verbose=True, monitor='kappa', mode='max')
lr_logger_callback = LearningRateLogger()
trainer = pl.Trainer(max_epochs=args.epochs,
                     gpus=[args.gpu],
                     precision=args.precision,
                     logger=logger,
                     checkpoint_callback=checkpoint_callback,
                     callbacks=[lr_logger_callback],
                     amp_level='O1' if args.precision == 16 else None)

model = EfficientNetV2(**vars(args), out_dim=5)
trainer.fit(model)
