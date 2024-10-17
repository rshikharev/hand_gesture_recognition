import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import configparser

# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

class FreiHANDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        :param root_dir: Корневая директория с данными FreiHAND.
        :param transform: Трансформации для изображений (например, Resize, ToTensor).
        :param mode: 'train' для тренировочных данных или 'eval' для данных оценки.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Загрузка аннотаций
        if self.mode == 'train':
            self.image_dir = os.path.join(self.root_dir, 'training')
            with open(os.path.join(self.root_dir, 'training_K.json'), 'r') as f:
                self.K_data = json.load(f)
            with open(os.path.join(self.root_dir, 'training_scale.json'), 'r') as f:
                self.scale_data = json.load(f)
            with open(os.path.join(self.root_dir, 'training_xyz.json'), 'r') as f:
                self.keypoints_data = json.load(f)
        elif self.mode == 'eval':
            self.image_dir = os.path.join(self.root_dir, 'evaluation')
            with open(os.path.join(self.root_dir, 'evaluation_K.json'), 'r') as f:
                self.K_data = json.load(f)
            with open(os.path.join(self.root_dir, 'evaluation_scale.json'), 'r') as f:
                self.scale_data = json.load(f)
        else:
            raise ValueError("Mode должен быть 'train' или 'eval'")

        # Загрузка списка файлов изображений
        self.image_paths = sorted([os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Загружаем изображение
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Применяем трансформации
        if self.transform:
            image = self.transform(image)

        # Загружаем аннотации
        if self.mode == 'train':
            keypoints = np.array(self.keypoints_data[idx])  # 3D координаты ключевых точек (21 x 3)
            keypoints = torch.tensor(keypoints, dtype=torch.float32)

            K = np.array(self.K_data[idx])  # Матрица камеры
            K = torch.tensor(K, dtype=torch.float32)

            scale = self.scale_data[idx]  # Масштаб руки
            scale = torch.tensor(scale, dtype=torch.float32)

            return image, keypoints, K, scale
        else:
            K = np.array(self.K_data[idx])  # Матрица камеры
            K = torch.tensor(K, dtype=torch.float32)

            scale = self.scale_data[idx]  # Масштаб руки
            scale = torch.tensor(scale, dtype=torch.float32)

            return image, K, scale
    
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