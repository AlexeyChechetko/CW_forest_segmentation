import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def count_pixels_by_color(path_to_image: str) -> dict:
    """
    Считаем количество пикселей разных цветов на маске

    :return:
        dict: Словарь с количеством пикселей по цветам на маске
    """
    image = cv2.imread(path_to_image)

    color_counts = {
        'red': 0,
        'yellow': 0,
        'green': 0,
        'black': 0
    }

    # Определим цвета в BGR (OpenCV использует BGR)
    colors_bgr = {
        'red':    ([55, 96, 255], [55, 96, 255]),
        'yellow': ([55, 250, 250], [55, 250, 250]),
        'green':  ([83, 179, 36], [83, 179, 36]),
        'black':  ([0, 0, 0], [0, 0, 0])
    }

    for color, (lower, upper) in colors_bgr.items():
        mask = cv2.inRange(image, np.array(lower), np.array(upper))
        count = cv2.countNonZero(mask)
        color_counts[color] = count

    return color_counts

def plot_color_histogram(color_counts: dict):
    colors = list(color_counts.keys())
    values = list(color_counts.values())

    # Настроим цвета столбцов так, чтобы совпадали с названиями
    bar_colors = {
        'red': 'red',
        'yellow': 'yellow',
        'green': 'green',
        'black': 'black'
    }

    plt.bar(colors, values, color=[bar_colors[c] for c in colors])
    plt.xlabel("Цвет")
    plt.ylabel("Количество пикселей")
    plt.title("Гистограмма по цветам маски")
    plt.show()


if __name__ == '__main__':
    path_to_mask = '../../data/mask.tif'

    colors = count_pixels_by_color(path_to_mask)
    print(colors)

    plot_color_histogram(colors)

    weights = {'red': 1, 'yellow': 3290532 / 1702202, 'green': 11578458 / 1702202, 'black': 47229678 / 1702202}
    tensor_weights = torch.tensor(list(weights.values()), dtype=torch.float32)
    print(weights)
    print(tensor_weights)
