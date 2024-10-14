import torch
import cv2
import configparser
from key_point import GestureClassifier, MediaPipeKeypointExtractor
import numpy as np
import mediapipe as mp

# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

# Получаем параметры из конфигурационного файла
model_save_path = config['Paths']['model_save_path']
num_keypoints = int(config['Model']['num_keypoints'])
num_classes = int(config['Model']['num_classes'])

# Словарь для отображения классов жестов
gesture_dict = {0: "like", 1: "dislike", 2: "hand_heart", 3: "ok", 4: "rock"}

def preprocess_image(image):
    return image

def draw_bounding_box(image, hand_landmarks):
    """
    Рисует прямоугольник вокруг руки, используя координаты ключевых точек.
    Возвращает координаты верхней левой точки (x_min, y_min).
    """
    h, w, _ = image.shape
    x_min, x_max = w, 0
    y_min, y_max = h, 0

    # Проходим по ключевым точкам, чтобы найти минимальные и максимальные координаты
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, x_max = min(x, x_min), max(x, x_max)
        y_min, y_max = min(y, y_min), max(y, y_max)

    # Рисуем прямоугольник вокруг руки
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return x_min, y_min  # Возвращаем координаты для текста

def run_inference(model, device):
    cap = cv2.VideoCapture(0)  # Открываем видеопоток с веб-камеры
    extractor = MediaPipeKeypointExtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image = preprocess_image(frame)
        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # Извлечение ключевых точек
        keypoints, hand_landmarks = extractor.extract_keypoints(image_rgb)
        if keypoints is not None:
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

            # Предсказание жеста
            with torch.no_grad():
                output = model(keypoints_tensor)
                _, predicted_class = torch.max(output, 1)

            # Получаем текстовое представление предсказанного жеста
            predicted_gesture = gesture_dict.get(predicted_class.item(), "Unknown")

            # Отрисовка прямоугольника вокруг руки и получение координат для текста
            x_min, y_min = draw_bounding_box(frame, hand_landmarks)

            # Отображение предсказанного жеста над рукой
            label = f"{predicted_gesture}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Отрисовка ключевых точек руки на кадре
            extractor.mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели с параметрами из конфига
    model = GestureClassifier(num_keypoints=num_keypoints, num_classes=num_classes)
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()

    run_inference(model, device)