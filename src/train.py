import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Класс Train для обучения моделей
class Train:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, num_epoch, device, loss_fn, path_to_results):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.device = device
        self.loss_fn = loss_fn
        self.train_losses = []
        self.val_losses = []
        self.path_to_results = path_to_results

    def train_step(self):
        self.model.to(self.device)
        self.model.train()
        total_loss = 0

        for images, masks in tqdm(self.train_dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)
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
                images = images.to(self.device)
                masks = masks.to(self.device)
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
        torch.save(self.model.state_dict(), self.path_to_results + '/trained-model')

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', color='red', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.path_to_results + '/loss.png')