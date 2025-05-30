import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from models.unet_model import UNet

# Класс Dataset для патчей
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask.astype(np.float32) / 255.0
        mask = np.transpose(mask, (2, 0, 1))

        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32)

        return image, mask

# Создание Dataset
train_dataset = SegmentationDataset('../data/train/images', '../data/train/labels')
val_dataset = SegmentationDataset('../data/val/images', '../data/val/labels')

# Создание DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

