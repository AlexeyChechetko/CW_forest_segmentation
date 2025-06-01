import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def create_dirs(base_dir: str) -> None:
    """
    Создает директории под train, test, val выборки, в которых будут храниться патчи

    :param base_dir: Базовая директория для патчей
    """
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test/labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val/labels'), exist_ok=True)

def get_image_array(path_to_image: str) -> np.ndarray:
    """
    Получаем изображение в виде матрицы формы (height, width, channels)

    :return изображение в числовом виде
    """
    return np.array(Image.open(path_to_image))

def slice_image(path_to_image: str, patch_shape: tuple[int, int]) -> list[np.ndarray]:
    """
    Нарезает исходное изображение на патчи со стороной по форме patch_shape

    :param patch_shape: форма патча (height, width)
    :param path_to_image: путь до нарезаемого изображения

    :return список патчей
    """
    image = get_image_array(path_to_image)

    patches = []

    height, width, channels = image.shape
    h_steps = (height + patch_shape[0] - 1) // patch_shape[0]
    w_steps = (width + patch_shape[1] - 1) // patch_shape[1]

    for h in range(h_steps):
        for w in range(w_steps):

            start_h = h * patch_shape[0]
            start_w = w * patch_shape[1]
            end_h = min(start_h + patch_shape[0], height)
            end_w = min(start_w + patch_shape[1], width)

            patch = image[start_h:end_h, start_w:end_w, :]

            if patch.shape[0] < patch_shape[0] or patch.shape[1] < patch_shape[1]:
                padded_patch = np.zeros(shape=(patch_shape[0], patch_shape[1], channels), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch

            patches.append(patch)

    return patches

def save_patches(patches: list, labels: list, split_type: str, base_dir: str, prefix: str) -> None:
    """
    После разделения на train, val и test выборки сохраняет патчи как .png изображения в нужные директории

    :param patches: патчи изображения
    :param labels: патчи маски
    :param split_type: train / val / test
    :param base_dir: директория с данными
    :param prefix: префикс в имени патча
    """
    for i, (patch, label) in enumerate(zip(patches, labels)):
        img = Image.fromarray(patch)
        img.save(os.path.join(base_dir, f'{split_type}/images/{prefix}_{i}.png'))

        label_img = Image.fromarray(label)
        label_img.save(os.path.join(base_dir, f'{split_type}/labels/{prefix}_{i}.png'))

if __name__ == '__main__':
    # Создаем директории для патчей из train, val и test выборок
    create_dirs('../../data/')

    # Нарезаем изображение и маску на патчи
    patches_image = slice_image(path_to_image='../../data/satellite_image.tif', patch_shape=(256, 256))
    patches_mask = slice_image(path_to_image='../../data/mask.tif', patch_shape=(256, 256))

    print(f"- Количество патчей размера 256*256: {len(patches_image)}")

    # Разделяем патчи на train, val и test выборки
    train_images, test_images, train_labels, test_labels = train_test_split(patches_image, patches_mask, test_size=0.2,
                                                                            random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5,
                                                                        random_state=42)

    print(f"- Размер train выборки: {len(train_images)}")
    print(f"- Размер val выборки: {len(val_images)}")
    print(f"- Размер test выборки: {len(test_images)}")

    # Сохраняем патчи
    save_patches(train_images, train_labels, 'train', '../../data/', 'patch')
    save_patches(val_images, val_labels, 'val', '../../data/', 'patch')
    save_patches(test_images, test_labels, 'test', '../../data/', 'patch')