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

        return label_image.astype(np.float32)


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

    def train(self):
        self.model.to(self.device)

        for epoch in range(self.num_epoch):
            # печатаем номер текущей эпохи
            print('* Epoch %d/%d' % (epoch+1, self.num_epoch))

            # 1. Обучаем сеть на картинках из train_loader
            self.model.train()  # train mode

            avg_train_loss = 0
            for i, (X_batch, Y_batch) in tqdm(enumerate(self.train_dataloader)):
                # переносим батч на GPU
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                # получаем ответы сети на батч
                Y_pred = self.model(X_batch)
                # print(Y_pred.shape)
                # print(Y_batch.shape)
                Y_batch = Y_batch.long()
                # считаем лосс, делаем шаг оптимизации сети
                loss = self.loss_fn(Y_pred, Y_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                avg_train_loss += loss / len(train_loader)

            # выводим средний лосс на тренировочной выборке за эпоху
            print('avg train loss: %f' % avg_train_loss)
            self.train_losses.append(avg_train_loss)

            # 2. Тестируем сеть на картинках из val_loader
            self.model.eval()

            avg_val_loss = 0
            for i, (X_batch, Y_batch) in enumerate(self.val_dataloader):
                # переносим батч на GPU
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                # получаем ответы сети на батч
                Y_pred = self.model(X_batch)
                # считаем лосс на батче
                Y_batch = Y_batch.long()
                loss = self.loss_fn(Y_pred, Y_batch)

                avg_val_loss += loss / len(val_loader)

            # выводим средний лосс на валидационных данных
            print('avg val loss: %f' % avg_val_loss)
            self.val_losses.append(avg_val_loss)



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

    # self, model, train_dataloader, val_dataloader, optimizer, num_epoch, device, loss_fn
    trainer = Train(unet_model, train_loader, val_loader, optim.Adam(unet_model.parameters(), lr=1e-3),
                       2, device, nn.CrossEntropyLoss())

    trainer.train()
    torch.save(unet_model.state_dict(), "../models/unet")




