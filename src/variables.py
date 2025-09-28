import torch

# Цвета маски
mask_label_map = {0: [0, 0, 0],  # black
                  1: [250, 250, 55],  # yellow
                  2: [36, 179, 83],  # green
                  3: [255, 96, 55]}  # red

# Относительные пути до train, val, test выборок
path_to_train_data = '../data/train'
path_to_val_data = '../data/val'
path_to_test_data = '../data/test'

# Проверка доступности GPU и device переменная
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CPU")

epochs = 30

# Относительные пути до папок, куда запишутся результаты train, test
path_to_train_results = '../results/train'
path_to_test_results = '../results/test'