import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import src.variables as variables


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

        all_predictions = all_predictions.flatten()
        all_labels = all_labels.flatten()

        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

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