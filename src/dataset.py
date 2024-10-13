import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class HandGestureDataset(Dataset):
    def __init__(self, dataset_path, class_to_idx=None, transform=None):
        """
        Args:
            dataset_path (str): Путь к директории с изображениями и метками жестов.
            class_to_idx (dict, optional): Словарь, где ключ - название жеста, значение - числовая метка.
            transform (callable, optional): Трансформации для изображений (например, аугментация).
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Создаем словарь class_to_idx, если его не передали
        if class_to_idx is None:
            self.class_to_idx = self._find_classes(dataset_path)
        else:
            self.class_to_idx = class_to_idx
        
        # Загрузка данных из директории
        self._load_data()

    def _find_classes(self, dataset_path):
        """Создает словарь соответствия названиям классов (жестов) числовым меткам."""
        classes = sorted(os.listdir(dataset_path))
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return class_to_idx

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
        
        # Применяем трансформации, если они указаны
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)