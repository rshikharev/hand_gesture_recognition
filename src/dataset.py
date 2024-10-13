import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import configparser

# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

class HandGestureDataset(Dataset):
    def __init__(self, dataset_path=None, class_to_idx=None, transform=None, image_size=(224, 224)):
        """
        Args:
            dataset_path (str): Путь к директории с изображениями и метками жестов.
            class_to_idx (dict, optional): Словарь, где ключ - название жеста, значение - числовая метка.
            transform (callable, optional): Трансформации для изображений (например, аугментация).
            image_size (tuple): Размер изображений, до которого они будут изменены.
        """
        # Если путь к данным не указан, берем его из конфига
        self.dataset_path = dataset_path if dataset_path else config['Paths']['train_data_path']
        self.image_size = image_size
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(self.image_size),  # Изменяем размер всех изображений до заданного размера
            transforms.ToTensor()
        ])

        # Получаем список классов из конфига
        class_names = config['Classes']['class_names'].split(', ')

        # Создаем словарь class_to_idx, если его не передали
        if class_to_idx is None:
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        else:
            self.class_to_idx = class_to_idx

        self.images = []
        self.labels = []

        # Загрузка данных из директории
        self._load_data()

    def _load_data(self):
        """Чтение изображений и меток жестов из папки."""
        for label_dir in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label_dir)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[label_dir])  # Метка жеста через словарь class_to_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Открываем изображение
        image = Image.open(img_path).convert("RGB")

        # Применяем трансформации (например, изменение размера и преобразование в тензор)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)