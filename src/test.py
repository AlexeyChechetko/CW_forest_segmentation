import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import variables as variables

def iou_per_class(y_true, y_pred, num_classes):
    """
    Считает IoU для каждого класса.
    y_true, y_pred : np.ndarray
        2D или 3D массивы (H,W) или (N,H,W)
    num_classes : int
        Количество классов (например 3)

    return: словарь {класс: IoU}
    """
    ious = {}
    for cls in range(num_classes):
        # создаем маски для текущего класса
        true_mask = (y_true == cls)
        pred_mask = (y_pred == cls)

        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()

        if union == 0:
            ious[cls] = float('nan')  # класс отсутствует
        else:
            ious[cls] = intersection / union
    return ious


def mean_iou(y_true, y_pred, num_classes):
    """
    Считает средний IoU по всем классам (mIoU).
    """
    ious = iou_per_class(y_true, y_pred, num_classes)
    values = [v for v in ious.values() if not np.isnan(v)]
    return np.mean(values), ious

def make_predictions(outputs):
    classes_prediction = nn.Softmax(dim=1)(outputs).squeeze()
    prediction = torch.argmax(classes_prediction, dim=1).long()

    return prediction

def make_picture_from_output(output):
    classes_prediction = nn.Softmax(dim=1)(output).squeeze()

    image_prediction = np.ndarray((classes_prediction.shape[1], classes_prediction.shape[2], 3), dtype=np.uint8)
    for i in range(classes_prediction.shape[1]):
        for j in range(classes_prediction.shape[2]):
            color_class = torch.argmax(classes_prediction[:, i, j]).item()
            image_prediction[i, j, :] = variables.mask_label_map[color_class]

    return image_prediction

def make_picture_from_mask(mask):
    image_mask = np.ndarray((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            image_mask[i, j, :] = variables.mask_label_map[mask[i, j].item()]

    return image_mask


def evaluate_segmentation(y_true, y_pred, ignore_class=0):
    """
    Оценка качества многоклассовой сегментации.
    Фон (ignore_class) исключается из метрик.

    Parameters
    ----------
    y_true : np.ndarray
        Истинные значения классов (H×W или 1D).
    y_pred : np.ndarray
        Предсказанные значения классов (H×W или 1D).
    ignore_class : int, optional
        Класс, который нужно игнорировать (по умолчанию 0).
    """
    # преобразуем в 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # фильтруем фон
    mask = y_true != ignore_class
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    classes = np.unique(y_true_f)
    target_names = [f"class_{c}" for c in classes]

    # precision, recall, f1
    precision = precision_score(y_true_f, y_pred_f, labels=classes, average="macro")
    recall = recall_score(y_true_f, y_pred_f, labels=classes, average="macro")
    f1 = f1_score(y_true_f, y_pred_f, labels=classes, average="macro")

    print("🔹 Метрики по классам (без фона):")
    print(classification_report(y_true_f, y_pred_f, labels=classes, target_names=target_names))

    print("🔹 Усреднённые метрики (macro):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    return precision, recall, f1

class Test:
    def __init__(self, model, test_dataset, test_dataloader, device, path_to_results):
        self.model = model
        self.test_dataset = test_dataset
        self.test_dataloader = test_dataloader
        self.device = device
        self.path_to_results = path_to_results

    def test(self, criterion):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, masks in tqdm(self.test_dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                masks = masks.long()

                outputs = self.model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item()

                predictions = make_predictions(outputs)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(masks.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        miou, ious = mean_iou(all_labels, all_predictions, num_classes=4)
        print("IoU по классам:", ious)
        print("Средний mIoU:", miou)

        precision, recall, f1 = evaluate_segmentation(all_labels, all_predictions, ignore_class=3)

        avg_loss = test_loss / len(self.test_dataset)

        with open(self.path_to_results + "/metrics.txt", "w") as f:
            f.write(f"avg_loss: {avg_loss:.4f}\n")
            f.write(f"precision: {precision:.4f}\n")
            f.write(f"recall: {recall:.4f}\n")
            f.write(f"f1: {f1:.4f}\n")

        return avg_loss, precision, recall, f1

    def visualize_segmentation_examples(self, start=0, num_examples=5):
        self.model.eval()

        with torch.no_grad():
            plt.figure(figsize=(15, num_examples * 5))
            for i in range(start, num_examples + start):
                image, mask = self.test_dataset[i]
                image = image.unsqueeze(0)

                output = self.model(image)

                image_prediction = make_picture_from_output(output)
                image_mask = make_picture_from_mask(mask)
                image = image.squeeze().numpy().transpose(1, 2, 0)

                plt.subplot(num_examples, 3, (i - start) * 3 + 1)
                plt.imshow(image)
                plt.title(f"Image {i + 1}")
                plt.axis("off")

                plt.subplot(num_examples, 3, (i - start) * 3 + 2)
                plt.imshow(image_mask)
                plt.title(f"Target {i + 1}")
                plt.axis("off")

                plt.subplot(num_examples, 3, (i - start) * 3 + 3)
                plt.imshow(image_prediction)
                plt.title(f"Prediction {i + 1}")
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(self.path_to_results + '/segmentation_examples' + str(start) + '_' + str(start+num_examples) + '.png')