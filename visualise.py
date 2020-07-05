import os

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

from tqdm import tqdm

root_path = f'/home/nvme/Kaggle/prostate-cancer-grade-assessment/'

df = pd.read_csv(root_path + 'train.csv')

mask_present = [] # Only about 100 images in the dataset have no mask so just ignore them for training
for idx in df['image_id']:
    mask_present += [os.path.isfile(os.path.join(root_path, 'train_label_masks', idx + '_mask.tiff'))]
df = df[mask_present]

dataset = PandaDataset(root_path, df, level=1, patch_size=256, num_patches=32, mode='train', use_mask=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=False, num_workers=4)

cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
for j in range(5):
    image, (mask, label) = dataset[j]
    plt.figure(figsize=(32, 32))

    for i in range(32):
        plt.subplot(8, 8, 2 * i + 1)
        plt.imshow(image[i])
        plt.subplot(8, 8, 2 * (i + 1))
        plt.imshow(mask[i], cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
