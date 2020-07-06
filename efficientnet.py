import time
import os

import numpy as np
import pandas as pd
import cv2
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

from albumentations import Compose, HorizontalFlip, VerticalFlip, Transpose

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from dataset import PandaDataset
from efficientnet_pytorch import model as enet
from loader import AsynchronousLoader
from scheduler import GradualWarmupScheduler

import matplotlib.pyplot as plt
plt.ion()

from tqdm import tqdm

class EfficientNetV2(nn.Module):
    def __init__(self, enet_type, pretrained_model, num_patches, out_dim):
        super(EfficientNetV2, self).__init__()
        self.enet = enet.EfficientNet.from_name(enet_type)
        self.enet.load_state_dict(torch.load(pretrained_model))

        self.fc_mixer = nn.Linear(num_patches, 1)
        self.fc_out = nn.Linear(self.enet._fc.in_features, out_dim)

    def extract(self, x):
        x = self.enet.extract_features(x)
        return F.adaptive_avg_pool2d(x, 1).squeeze()

    def forward(self, x):
        batch_size, num_patches, height, width, channels = x.shape

        # Apply a separate identical enet on every separate patch
        x = x.view(batch_size * num_patches, height, width, channels)
        x = self.extract(x).view(batch_size, num_patches, -1)

        x = self.fc_mixer(x.permute(0, 2, 1)).view(batch_size, -1)
        x = self.fc_out(x)
        return x

root_path = f'/home/nvme/Kaggle/prostate-cancer-grade-assessment/'

enet_type = 'efficientnet-b0'
pretrained_model = f'EfficientNet-PyTorch/efficientnet-b0-08094119.pth'

epochs = 10
batch_size = 2
num_workers = 16
lr = 3e-4
warmup_factor = 10
warmup_epo = 1

num_patches = 20
patch_size = 256
level = 1

df = pd.read_csv(root_path + 'train.csv')

mask_present = [] # Only about 100 images in the dataset have no mask so just ignore them for training
for idx in df['image_id']:
    mask_present += [os.path.isfile(os.path.join(root_path, 'train_label_masks', idx + '_mask.tiff'))]
df = df[mask_present]

train_df, validation_df = train_test_split(df)

train_set = PandaDataset(root_path, train_df, level=level, patch_size=patch_size, num_patches=num_patches, mode='train', use_mask=False)
validation_set = PandaDataset(root_path, validation_df, level=level, patch_size=patch_size, num_patches=num_patches, mode='train', use_mask=False)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)

train_dataloader = AsynchronousLoader(train_dataloader, device=torch.device('cuda', 0), q_size=5)
validation_dataloader = AsynchronousLoader(validation_dataloader, device=torch.device('cuda', 0), q_size=5)

model = EfficientNetV2(enet_type=enet_type, pretrained_model=pretrained_model, num_patches=num_patches, out_dim=5)
model = model.cuda(0)

optim = optimizer = optim.Adam(model.parameters(), lr=lr/warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup_epo)
scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

criterion = nn.BCEWithLogitsLoss()

best_kappa = 0
for i in range(epochs):
    print('\nTraining epoch', i)
    model.train()
    pb = tqdm(train_dataloader, total=len(train_dataloader))
    avg_loss = 0
    for x, y in pb:
        optim.zero_grad()
        logits = model(x.float())
        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        avg_loss = 0.1 * loss.item() + 0.9 * avg_loss
        pb.update(1)
        pb.set_postfix(loss=avg_loss)
    pb.close()
    scheduler.step()


    print('Validating epoch', i)
    model.eval()
    pb = tqdm(validation_dataloader, total=len(train_dataloader))
    logits = []
    labels = []
    for x, y in pb:
        logits += [model(x.float()).detach().cpu().numpy()]
        labels += [y.detach().cpu().numpy()]
        pb.update(1)
    pb.close()

    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    out = 1.0 / (1.0 + np.exp(-logits))
    loss = -np.mean(labels * np.log(out) + (1 - labels) * np.log(1 - out))
    kappa = cohen_kappa_score(np.sum(out, axis=-1).round(), labels.sum(out_axis=-1))

    print('Validation loss {} and kappa {}'.format(loss, kappa))

    if kappa > best_kappa:
        print('Current kappa greater than previous best kappa of {}. Saving model'.format(best_kappa))
        best_kappa = kappa
        torch.save(model.state_dict(), 'efficientnetv2.pth')
