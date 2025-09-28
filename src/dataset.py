import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import re
import variables as variables


def colormap_to_labelmap(mask):
    label_image = np.zeros_like(mask[:, :, 0], dtype=np.uint8)

    for label, color in variables.mask_label_map.items():
        color_array = np.array(color)
        mask_condition = np.all(mask == color_array, axis=-1)
        label_image[mask_condition] = label

    return label_image

# Класс Dataset для патчей
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir), key=lambda x: int(re.search(r'\d+', x).group()))
        self.mask_paths = sorted(os.listdir(mask_dir), key=lambda x: int(re.search(r'\d+', x).group()))
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
        mask = colormap_to_labelmap(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.tensor(mask, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32)

        return image, mask, self.mask_paths[idx]