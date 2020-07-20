import os
import time

# There are two ways to load the data from the PANDA dataset:
# Option 1: Load images using openslide
import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
import random
import seaborn as sns
import cv2

# General packages
import pandas as pd
import numpy as np
from scipy import ndimage

import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import PIL

from torch.utils.data import DataLoader
from dataset import PandaDataset

from albumentations import Compose, HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue, RandomBrightness, RandomContrast, RandomGamma, ShiftScaleRotate

from tqdm import tqdm

root_path = f'/home/nvme/Kaggle/prostate-cancer-grade-assessment/'

df = pd.read_csv(root_path + 'train.csv')

mask_present = [] # Only about 100 images in the dataset have no mask so just ignore them for training
for idx in df['image_id']:
    mask_present += [os.path.isfile(os.path.join(root_path, 'train_label_masks', idx + '_mask.tiff'))]
df = df[mask_present]

transforms = Compose([Transpose(p=0.5),
                      VerticalFlip(p=0.5),
                      HorizontalFlip(p=0.5),
                      #ShiftScaleRotate(p=0.5)
                      ])
dataset = PandaDataset(root_path, df, level=1, patch_size=256, num_patches=25, use_mask=True, transforms=transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=16)

t0 = time.time()
x, y = dataset[2209]
t1 = time.time()
print(t1 - t0)
x, y = dataset[0]
t2 = time.time()
print(t2 - t1)
print(y[0].shape)

#for x, y, in tqdm(dataloader, total=len(dataloader)):
    #pass

cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'blue', 'green', 'yellow', 'red'])
for j in range(5):
    t0 = time.time()
    image, (mask, label) = dataset[j]
    print('Dataloading time', time.time() - t0)
    plt.figure(figsize=(32, 32))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(mask.max(dim=0)[1], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
