import cv2

cap = cv2.VideoCapture(0)  # Открываем видеопоток с веб-камеры

if not cap.isOpened():
    print("Ошибка: Не удалось открыть веб-камеру")
else:
    print("Веб-камера успешно подключена")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр")
        break

    # Отображаем видеопоток
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажми 'q', чтобы выйти
        break

cap.release()
cv2.destroyAllWindows()