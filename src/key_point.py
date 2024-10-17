import configparser
import torch
import mediapipe as mp
import cv2
import torch.nn as nn
import numpy as np
from torchvision import transforms


# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

# Извлекаем параметры модели из конфигурационного файла
num_keypoints = int(config['Model']['num_keypoints'])
num_classes = int(config['Model']['num_classes'])
class MediaPipeKeypointExtractor:
    """Используем MediaPipe для извлечения ключевых точек из изображений."""
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
    def extract_keypoints(self, image):
        """
        Извлекаем ключевые точки для изображения руки.
        Возвращаем 21 ключевую точку (x, y) или None, если рука не была обнаружена.
        """
        # Проверяем, является ли изображение тензором, если да — преобразуем его в numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # Преобразуем тензор в numpy, меняем порядок осей с [C, H, W] на [H, W, C]
        # Приводим изображение в формат uint8
        image = (image * 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(image_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y])
            return np.array(keypoints).flatten()  # Возвращаем плоский массив ключевых точек
        else:
            return None

        
class GestureClassifier(nn.Module):
    def __init__(self, num_keypoints=num_keypoints, num_classes=num_classes, dropout_rate=0.3, l2_reg=0.01):
        super(GestureClassifier, self).__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_keypoints * 2, 512),
            nn.LayerNorm(512),  # Добавляем LayerNorm для ускорения сходимости
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Добавляем LayerNorm для ускорения сходимости
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, keypoints):
        """
        Входные данные: ключевые точки, извлечённые из MediaPipe.
        Размер входа: (batch_size, num_keypoints * 2) — для каждого ключевого положения (x, y).
        """
        return self.fc_block(keypoints)
    
class CustomKeypointExtractor:
    """Используем обученную модель для извлечения ключевых точек из изображений."""
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Загружаем обученную модель
        self.device = device
        self.model = torch.load(model_path)
        self.model.eval()
        self.model.to(self.device)

        # Трансформации для изображения
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Замените на нужный размер
            transforms.ToTensor(),
        ])

    def extract_keypoints(self, image):
        """
        Извлекаем ключевые точки для изображения руки.
        Возвращаем 21 ключевую точку (x, y) или None, если рука не была обнаружена.
        """
        # Проверяем, является ли изображение тензором, если да — преобразуем его в numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # Преобразуем тензор в numpy, меняем порядок осей с [C, H, W] на [H, W, C]

        # Приводим изображение в формат uint8
        image = (image * 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Преобразуем изображение для модели
        input_image = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Предсказание ключевых точек
        with torch.no_grad():
            keypoints = self.model(input_image)
        
        keypoints = keypoints.cpu().numpy().reshape(-1, 2)  # Преобразуем в массив (21, 2)

        return keypoints.flatten()  # Возвращаем плоский массив ключевых точек (x, y)