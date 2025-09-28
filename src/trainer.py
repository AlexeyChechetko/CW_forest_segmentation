from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from unet_model import UNet
from dataset import SegmentationDataset
import variables as variables
from train import Train
from test_weight_func import compute_weight_map

if __name__ == '__main__':

    # Создание Dataset
    train_dataset = SegmentationDataset(variables.path_to_train_data + '/images', variables.path_to_train_data + '/labels')
    val_dataset = SegmentationDataset(variables.path_to_val_data + '/images', variables.path_to_val_data + '/labels')

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Создание модели
    unet_model = UNet()

    weights = torch.tensor([ 1.0000,  1 / 1.9331,  1 / 6.8020, 1 / 27.7462]).to(variables.device)

    # Обучение модели, сохранение модели, сохранение графика losses
    trainer = Train(unet_model, train_loader, val_loader, optim.Adam(unet_model.parameters(), lr=1e-3),
                    variables.epochs, variables.device, nn.CrossEntropyLoss(weight=weights, reduction='none'), variables.path_to_train_results, compute_weight_map)

    trainer.train()
    trainer.save_model()
    trainer.plot_losses()




