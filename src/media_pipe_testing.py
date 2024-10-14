import cv2
import mediapipe as mp

# Инициализация MediaPipe для работы с руками
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Захват видео с камеры
cap = cv2.VideoCapture(0)

# Инициализация модели для детекции рук
with mp_hands.Hands(
    static_image_mode=False,        # Детекция на каждом кадре
    max_num_hands=2,                # Максимум две руки
    min_detection_confidence=0.5,   # Минимальная вероятность детекции
    min_tracking_confidence=0.5     # Минимальная вероятность трекинга
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование изображения в RGB, так как MediaPipe работает с RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обработка изображения MediaPipe
        results = hands.process(image_rgb)
        
        # Если руки найдены
        if results.multi_hand_landmarks:
            # Проходим по каждой найденной руке
            for hand_landmarks in results.multi_hand_landmarks:
                # Отрисовка ключевых точек и соединений на изображении
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Отображаем результат
        cv2.imshow('MediaPipe Hand Tracking', frame)

        # Нажмите 'q' для выхода из программы
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
