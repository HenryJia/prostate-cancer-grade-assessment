import os
import time

# There are two ways to load the data from the PANDA dataset:
# Option 1: Load images using openslide
import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
from skimage.io import MultiImage
from skimage.measure import block_reduce
import random
import seaborn as sns
import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

class PandaDataset(Dataset):
    def __init__(self, root_path, df, level=2, num_patches=5, patch_size=256, use_mask=False, transforms=None):
        self.root_path = root_path
        self.df = df
        self.level = level
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.use_mask = use_mask
        self.transforms = transforms

        # Note: On the training set, the spacings have a mean and stddev of 0.47919909031475766 and 0.0192301637137064 so we needn't worry about them
        #spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

    def __getitem__(self, idx):
        path = os.path.join(self.root_path, 'train_images')
        # Skimage seems to be slightly faster
        #image = openslide.OpenSlide(os.path.join(path, self.df['image_id'].iloc[idx] + '.tiff'))
        image = MultiImage(os.path.join(path, self.df['image_id'].iloc[idx] + '.tiff'), conserve_memory=False)[self.level]

        #image = np.array(image.read_region((0, 0), self.level, image.level_dimensions[self.level]))

        # Only look at regions of the image that aren't empty space and put a bounding box on it
        # Find those regions using a subsampled image, since NumPy is slow
        stride = self.patch_size // 8
        f_blank = lambda x, axis: np.mean((x - 255) ** 2, axis=axis) * np.var(x, axis=axis)
        proportion_blank = block_reduce(image[::stride, ::stride], block_size=(self.patch_size // stride, self.patch_size // stride, 3), func=f_blank)


        regions = np.argsort(proportion_blank, axis=None)[::-1]
        x = regions % proportion_blank.shape[1] * self.patch_size
        y = regions // proportion_blank.shape[1] * self.patch_size

        patches = np.full((self.num_patches, self.patch_size, self.patch_size, 3), 255, dtype=np.uint8)
        for i in range(min(self.num_patches, x.shape[0])):
            img = image[y[i]:y[i] + self.patch_size, x[i]:x[i] + self.patch_size]
            patches[i, :img.shape[0], :img.shape[1]] = img
        image = patches

        label = torch.zeros(5)
        label[:self.df['isup_grade'].iloc[idx]] = 1

        if self.use_mask:
            #mask = openslide.OpenSlide(os.path.join(self.root_path, 'train_label_masks', self.df['image_id'].iloc[idx] + '_mask.tiff'))
            mask = MultiImage(os.path.join(self.root_path, 'train_label_masks', self.df['image_id'].iloc[idx] + '_mask.tiff'), conserve_memory=False)[self.level]
            mask = mask[..., 0]

            mask_patches = np.zeros((self.num_patches, self.patch_size, self.patch_size), dtype=np.uint8)
            for i in range(min(self.num_patches, x.shape[0])):
                msk = mask[y[i]:y[i] + self.patch_size, x[i]:x[i] + self.patch_size]
                mask_patches[i, :msk.shape[0], :msk.shape[1]] = msk
            mask = mask_patches

            if self.df['data_provider'].iloc[idx] == 'karolinska': # Different data providers have different mask formats, normalise them to be the same
                mask[mask == 2] = 3
                mask[mask == 1] = 2

            if self.transforms:
                for i in range(self.num_patches): # We need to iterate and apply to each image separately
                    augmented = self.transforms(image=image[i], mask=mask[i])
                    image[i] = augmented['image']
                    mask[i] = augmented['mask']

            # Convert our mask to binned binary just like the labels
            mask_binary = np.zeros((mask.shape[0], 6, mask.shape[1], mask.shape[2]))
            for i in range(6):
                mask_binary[:, i] = (i == mask)
            mask = mask_binary

            #n = int(np.sqrt(self.num_patches))
            #image = image.reshape(n, n, self.patch_size, self.patch_size, 3).transpose((0, 2, 1, 3, 4)).reshape(n * self.patch_size, n * self.patch_size, 3)
            #mask = mask.reshape(n, n, self.patch_size, self.patch_size, 6).transpose((0, 2, 1, 3, 4)).reshape(n * self.patch_size, n * self.patch_size, 6)

            return torch.tensor(image).permute(0, 3, 1, 2), (torch.tensor(mask), label)

        if self.transforms:
            for i in range(self.num_patches): # We need to iterate and apply to each image separately
                image[i] = self.transforms(image=image[i])['image']

        #n = int(np.sqrt(self.num_patches))
        #image = image.reshape(n, n, self.patch_size, self.patch_size, 3).transpose((0, 2, 1, 3, 4)).reshape(n * self.patch_size, n * self.patch_size, 6)

        return torch.tensor(image).permute(0, 3, 1, 2), label



    def __len__(self):
        return len(self.df)
