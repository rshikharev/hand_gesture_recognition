import configparser
import torch

import mediapipe as mp
import cv2
import torch.nn as nn
import numpy as np

# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

# Извлекаем параметры модели из конфигурационного файла
num_keypoints = int(config['Model']['num_keypoints'])
num_classes = int(config['Model']['num_classes'])

class MediaPipeKeypointExtractor:
    """Используем MediaPipe для извлечения ключевых точек из изображений."""
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_keypoints(self, image):
        """
        Извлекаем ключевые точки для изображения руки.
        Возвращаем 21 ключевую точку (x, y) и landmarks (для отображения),
        или None, если рука не была обнаружена.
        """
        # Проверяем, является ли изображение тензором, если да — преобразуем его в numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # Преобразуем тензор в numpy, меняем порядок осей с [C, H, W] на [H, W, C]

        # Приводим изображение в формат uint8
        image = (image * 255).astype(np.uint8)

        # Преобразуем изображение в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Используем MediaPipe для обработки изображения
        result = self.mp_hands.process(image_rgb)

        # Если найдены landmarks (ключевые точки рук)
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]  # Берем первую найденную руку
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y])  # Извлекаем координаты ключевых точек (x, y)
            
            # Возвращаем массив ключевых точек и сами landmarks
            return np.array(keypoints).flatten(), hand_landmarks
        else:
            # Если рука не была обнаружена, возвращаем None
            return None, None

        
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