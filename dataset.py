import os
import time

# There are two ways to load the data from the PANDA dataset:
# Option 1: Load images using openslide
import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
from skimage.io import MultiImage
import random
import seaborn as sns
import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

class PandaDataset(Dataset):
    def __init__(self, root_path, df, level=2, num_patches=5, patch_size=256, num_trials=10, mode='train', use_mask=False, transforms=None):
        self.root_path = root_path
        self.df = df
        self.level = level
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_trials = num_trials
        self.mode = mode
        self.use_mask = use_mask
        self.transforms = transforms

        # Note: On the training set, the spacings have a mean and stddev of 0.47919909031475766 and 0.0192301637137064 so we needn't worry about them
        #spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

    def __getitem__(self, idx):
        path = os.path.join(self.root_path, 'train_images') if self.mode == 'train' else os.path.join(self.root_path, 'test_images')
        # Skimage seems to be slightly faster
        #image = openslide.OpenSlide(os.path.join(path, self.df['image_id'].iloc[idx] + '.tiff'))
        image = MultiImage(os.path.join(path, self.df['image_id'].iloc[idx] + '.tiff'), conserve_memory=False)[self.level]


        if self.num_patches:
            #image = np.array(image.read_region((0, 0), self.level, image.level_dimensions[self.level]))

            # Only look at regions of the image that aren't empty space and put a bounding box on it
            # Find those regions using a subsampled image, since NumPy is slow
            stride = 32
            regions = np.stack(np.where(np.all(image[::stride, ::stride] != 255, axis=-1)), axis=1) * stride
            if len(regions) == 0: # In case we have a completely blank image
                regions = np.array(((0, 0, 0), image.shape))
            start = np.min(regions, axis=0)
            end = np.max(regions, axis=0)
            shape = end - start

            # Pad to make our image divisible by our patch size
            image = image[start[0]:end[0], start[1]:end[1]]
            image = np.pad(image, ((0, self.patch_size - shape[0] % self.patch_size), (0, self.patch_size - shape[1] % self.patch_size), (0, 0)), constant_values=255)

            image = image.reshape(image.shape[0] // self.patch_size, self.patch_size, int(image.shape[1] / self.patch_size), self.patch_size, 3)
            image = image.transpose(0, 2, 1, 3, 4).reshape(-1, self.patch_size, self.patch_size, 3)

            if image.shape[0] < self.num_patches: # Pad up to reach the number of desired patches if we don't have enough
                image = np.pad(image, ((0, self.num_patches - image.shape[0]), (0, 0), (0, 0), (0, 0)), constant_values=255)
            proportion_blank = np.mean((image - 255) ** 2, axis=(1, 2, 3))

            selected_patches = np.argsort(proportion_blank)[-self.num_patches:]
            image = np.transpose(image[selected_patches], (0, 3, 1, 2))

        if self.mode == 'train':
            label = torch.zeros(5)
            label[:self.df['isup_grade'].iloc[idx]] = 1

            if self.use_mask:
                #mask = openslide.OpenSlide(os.path.join(self.root_path, 'train_label_masks', self.df['image_id'].iloc[idx] + '_mask.tiff'))
                mask = MultiImage(os.path.join(self.root_path, 'train_label_masks', self.df['image_id'].iloc[idx] + '_mask.tiff'), conserve_memory=False)[self.level]
                mask = mask[start[0]:end[0], start[1]:end[1], 0]

                if self.num_patches: # Apply the identical operations to the masks
                    #mask = np.array(mask.read_region((0, 0), self.level, mask.level_dimensions[self.level]))[start[0]:end[0], start[1]:end[1], 0]
                    mask = np.pad(mask, ((0, self.patch_size - shape[0] % self.patch_size), (0, self.patch_size - shape[1] % self.patch_size)), constant_values=0)
                    mask = mask.reshape(mask.shape[0] // self.patch_size, self.patch_size, mask.shape[1] // self.patch_size, self.patch_size)
                    mask = mask.transpose(0, 2, 1, 3).reshape(-1, self.patch_size, self.patch_size)

                    if mask.shape[0] < self.num_patches:
                        mask = np.pad(mask, ((0, self.num_patches - mask.shape[0]), (0, 0), (0, 0)))
                    mask = mask[selected_patches]

                    if self.df['data_provider'].iloc[idx] == 'radboud': # Different data providers have different mask formats, normalise them to be the same
                        mask[mask == 2] = 1
                        mask[mask > 2] = 2

                if self.transforms:
                    augmented = self.transforms(image=image, mask=mask)
                    image, mask = augmented['image'], augmented['mask']
                return torch.tensor(image), (torch.tensor(mask), label)

            if self.transforms:
                image = self.transforms(image=image)['image']
            return torch.tensor(image), label

        return torch.tensor(image)


    def __len__(self):
        return len(self.df)
