import torch
import cv2
import configparser
from key_point import GestureClassifier, MediaPipeKeypointExtractor
import numpy as np

# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

# Получаем параметры из конфигурационного файла
model_save_path = config['Paths']['model_save_path']
num_keypoints = int(config['Model']['num_keypoints'])
num_classes = int(config['Model']['num_classes'])

def preprocess_image(image):
    # Здесь можно добавить предобработку изображения, если это необходимо.
    return image

def run_inference(model, device):
    cap = cv2.VideoCapture(0)  # Открываем видеопоток с веб-камеры
    extractor = MediaPipeKeypointExtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image = preprocess_image(frame)

        # Извлечение ключевых точек
        keypoints = extractor.extract_keypoints(input_image)
        if keypoints is not None:
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

            # Предсказание жеста
            with torch.no_grad():
                output = model(keypoints_tensor)
                _, predicted_class = torch.max(output, 1)

            # Отображение результата
            label = f"Predicted Gesture: {predicted_class.item()}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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