import torch
import cv2
import configparser
from key_point import GestureClassifier, CustomKeypointExtractor  # Используем ваш CustomKeypointExtractor

# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

# Получаем параметры из конфигурационного файла
model_save_path = config['Paths']['model_save_path']
keypoint_model_path = config['Paths']['hourglass_model_save_path']  # Путь к модели ключевых точек
num_keypoints = int(config['Model']['num_keypoints'])
num_classes = int(config['Model']['num_classes'])

# Словарь для отображения классов жестов
gesture_dict = {0: "like", 1: "dislike", 2: "hand_heart", 3: "ok", 4: "rock"}

def preprocess_image(image):
    return image

def draw_bounding_box(image, keypoints):
    """
    Рисует прямоугольник вокруг руки, используя предсказанные ключевые точки.
    Возвращает координаты верхней левой точки (x_min, y_min).
    """
    h, w, _ = image.shape
    keypoints = keypoints.reshape(-1, 2)  # Преобразуем в массив пар (x, y)
    
    x_min, x_max = w, 0
    y_min, y_max = h, 0

    # Проходим по ключевым точкам, чтобы найти минимальные и максимальные координаты
    for keypoint in keypoints:
        x, y = int(keypoint[0] * w), int(keypoint[1] * h)  # Преобразуем относительные координаты в абсолютные
        x_min, x_max = min(x, x_min), max(x, x_max)
        y_min, y_max = min(y, y_min), max(y, y_max)

    # Рисуем прямоугольник вокруг руки
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return x_min, y_min  # Возвращаем координаты для текста

def run_inference(model, keypoint_extractor, device):
    cap = cv2.VideoCapture(0)  # Открываем видеопоток с веб-камеры

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image = preprocess_image(frame)
        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Извлечение ключевых точек с помощью вашей обученной модели
        keypoints = keypoint_extractor.extract_keypoints(image_rgb)
        
        if keypoints is not None:
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

            # Предсказание жеста
            with torch.no_grad():
                output = model(keypoints_tensor)
                _, predicted_class = torch.max(output, 1)

            # Получаем текстовое представление предсказанного жеста
            predicted_gesture = gesture_dict.get(predicted_class.item(), "Unknown")

            # Отрисовка прямоугольника вокруг руки и получение координат для текста
            x_min, y_min = draw_bounding_box(frame, keypoints)

            # Отображение предсказанного жеста над рукой
            label = f"{predicted_gesture}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели для распознавания жестов
    model = GestureClassifier(num_keypoints=num_keypoints, num_classes=num_classes)
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()

    # Загрузка модели для предсказания ключевых точек (например, HourglassNetwork)
    keypoint_extractor = CustomKeypointExtractor(keypoint_model_path, device=device)

    run_inference(model, keypoint_extractor, device)
