import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from models.unet_model import UNet
import matplotlib.pyplot as plt

# Класс Dataset для патчей
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_map, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.label_map = label_map
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
        mask = self.colormap_to_labelmap(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.tensor(mask, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32)

        return image, mask

    def colormap_to_labelmap(self, mask):
        label_image = np.zeros_like(mask[:, :, 0], dtype=np.uint8)

        for label, color in self.label_map.items():
            color_array = np.array(color)
            mask_condition = np.all(mask == color_array, axis=-1)
            label_image[mask_condition] = label

        return label_image #.astype(np.float32)


class Train:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, num_epoch, device, loss_fn):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.device = device
        self.loss_fn = loss_fn
        self.train_losses = []
        self.val_losses = []

    def train_step(self):
        self.model.to(self.device)
        self.model.train()
        total_loss = 0

        for images, masks in tqdm(self.train_dataloader):
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.long()
            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.loss_fn(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def validate_step(self):
        self.model.to(self.device)
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(self.val_dataloader):
                images = images.to(device)
                masks = masks.to(device)
                masks = masks.long()

                outputs = self.model(images)

                loss = self.loss_fn(outputs, masks)
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def train(self):
        for epoch in range(self.num_epoch):
            print(f"[INFO] Эпоха: {epoch + 1}/{self.num_epoch}")

            train_loss = self.train_step()
            self.train_losses.append(train_loss)

            val_loss = self.validate_step()
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def save_model(self):
        torch.save(self.model.state_dict(), "trained-model")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', color='red', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('Train_and_Validation_Losses.png')




if __name__ == '__main__':
    # Цвета маски
    mask_label_map = {0: [55, 96, 255],  # red
                      1: [55, 250, 250],  # yellow
                      2: [83, 179, 36],  # green
                      3: [0, 0, 0]}  # black

    # Создание Dataset
    train_dataset = SegmentationDataset('../data/train/images', '../data/train/labels', mask_label_map)
    val_dataset = SegmentationDataset('../data/val/images', '../data/val/labels', mask_label_map)

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Создание модели
    unet_model = UNet()

    # Проверка доступности GPU и device переменная
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CPU")

    # Обучение модели, сохранение модели, сохранение графика losses
    trainer = Train(unet_model, train_loader, val_loader, optim.Adam(unet_model.parameters(), lr=1e-3),
                       5, device, nn.CrossEntropyLoss())

    trainer.train()
    trainer.save_model()
    trainer.plot_losses()
    # test 1




