import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset

import os
from glob import glob
import tifffile as tiff
import numpy as np


class ClinicDataset(Dataset):
    def __init__(self):
        super(ClinicDataset, self).__init__()

        self.input_path = './Clinic_dataset/input_image'
        self.label_path = './Clinic_dataset/binary_image'
        self.input_files = sorted(glob(os.path.join(self.input_path, '*.tif')))
        self.label_files = sorted(glob(os.path.join(self.label_path, '*.tif')))

        self.input_transform = transform.Compose([
            transform.ToTensor()
        ])
        self.label_transform = transform.Compose([
            transform.ToTensor()
        ])

    def __getitem__(self, index):
        input_numpy = tiff.imread(self.input_files[index])
        label_numpy = tiff.imread(self.label_files[index])
        label_numpy = np.expand_dims(label_numpy, axis=2)
        input_torch = self.input_transform(input_numpy)
        label_torch = self.label_transform(label_numpy)
        return input_torch, label_torch

    def __len__(self):
        return len(self.input_files)
