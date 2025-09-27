from unet_model import UNet
from dataset import SegmentationDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import variables as variables
from test import Test

if __name__ == '__main__':
    # Путь к сохраненной модели
    model_path = variables.path_to_train_results + "/trained-model"

    # Загрузка модели (пример для UNet или другой твоей модели)
    unet_model = UNet()  # ← создай модель той же архитектуры
    unet_model.load_state_dict(torch.load(model_path))

    test_dataset = SegmentationDataset('../data/test/images', '../data/test/labels')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    tester = Test(unet_model, test_dataset, test_loader, variables.device, variables.path_to_test_results)
    tester.visualize_segmentation_examples(0, 5)
    tester.visualize_segmentation_examples(20, 5)
    tester.visualize_segmentation_examples(50, 5)
    tester.visualize_segmentation_examples(30, 5)
    test_loss, test_precision, test_recall, test_f1 = tester.test(nn.CrossEntropyLoss())
    print(f"Test Loss: {test_loss:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}")
