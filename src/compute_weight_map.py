import numpy as np
from scipy import ndimage
import variables as variables
from dataset import SegmentationDataset
import torch

def compute_weight_map(mask, wc=None, w0=10, sigma=5):
    """
    mask: np.array HxW (целые метки классов, 0 = фон, 1..N = объекты)
    wc: np.array (num_classes,) - веса классов (например, inverse frequency)
    w0, sigma: параметры из статьи

    return: np.array HxW (weight map)
    """
    # (1) базовый вес по классам
    if wc is None:
        wc = np.ones(int(mask.max()) + 1)
    weight = wc[mask]

    # (2) находим объекты (компоненты связности)
    obj_mask = mask > 0
    labeled, n_objs = ndimage.label(obj_mask)

    if n_objs < 2:
        return weight  # если всего один объект → нечего разделять

    # (3) для каждого объекта считаем distance transform
    distances = []
    for i in range(1, n_objs + 1):
        obj_i = labeled == i
        dist_map = ndimage.distance_transform_edt(~obj_i)
        distances.append(dist_map)

    distances = np.stack(distances, axis=0)  # shape (n_objs, H, W)

    # (4) сортируем расстояния, берём два наименьших
    d_sorted = np.sort(distances, axis=0)
    d1, d2 = d_sorted[0], d_sorted[1]  # ближайший и второй ближайший

    # (5) формула веса
    boundary_term = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))

    return weight + boundary_term

def compute_maps(train_dataset, val_dataset):
    # Словарь карт весов, где key = 'patch_x.png', а value = np.array([weight_map])
    train_weight_map = {}
    val_weight_map = {}

    # Создаем для train_dataset
    for image, _mask, mask_name in train_dataset:
        train_weight_map[mask_name] = torch.from_numpy(compute_weight_map(mask=_mask.numpy().astype(np.int64), wc=variables.weights,
                                                         w0=3, sigma=5)).to(variables.device)
    # Создаем для val_dataset
    for image, _mask, mask_name in val_dataset:
        val_weight_map[mask_name] = torch.from_numpy(compute_weight_map(mask=_mask.numpy().astype(np.int64), wc=variables.weights, w0=3,
                                                       sigma=5)).to(variables.device)

    # Можно сохранить в текстовый файл карты для проверки
    # np.savetxt("patch_0.txt", weight_map['patch_1.png'], fmt="%f")

    return train_weight_map, val_weight_map

if __name__ == "__main__":
    # Создание Dataset
    _train_dataset = SegmentationDataset(variables.path_to_train_data + '/images', variables.path_to_train_data + '/labels')
    _val_dataset = SegmentationDataset(variables.path_to_val_data + '/images', variables.path_to_val_data + '/labels')

    _train_weight_map, _val_weight_map = compute_maps(_train_dataset, _val_dataset)