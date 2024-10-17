import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Функция для разделения данных на train и val.
    
    Args:
        source_dir (str): Путь к исходной папке с данными.
        train_dir (str): Путь к папке для тренировочных данных.
        val_dir (str): Путь к папке для валидационных данных.
        split_ratio (float): Соотношение разделения данных на train и val (по умолчанию 0.8).
    """
    # Проходим по каждой папке (каждый жест)
    for label_dir in os.listdir(source_dir):
        label_path = os.path.join(source_dir, label_dir)
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            random.shuffle(images)  # Перемешиваем список изображений

            # Определяем индекс разделения
            split_index = int(len(images) * split_ratio)
            train_images = images[:split_index]
            val_images = images[split_index:]

            # Создаём папки для train и val, если их нет
            train_label_dir = os.path.join(train_dir, label_dir)
            val_label_dir = os.path.join(val_dir, label_dir)
            os.makedirs(train_label_dir, exist_ok=True)
            os.makedirs(val_label_dir, exist_ok=True)

            # Перемещаем изображения в соответствующие папки
            for img in train_images:
                shutil.move(os.path.join(label_path, img), os.path.join(train_label_dir, img))

            for img in val_images:
                shutil.move(os.path.join(label_path, img), os.path.join(val_label_dir, img))

            print(f"Жест '{label_dir}': {len(train_images)} в train, {len(val_images)} в val.")

if __name__ == "__main__":
    # Указываем пути к папкам
    source_dir = "data/raw"       # Папка с исходными изображениями
    train_dir = "data/train"      # Папка для тренировочных данных
    val_dir = "data/val"          # Папка для валидационных данных

    # Вызываем функцию для разделения данных
    split_data(source_dir, train_dir, val_dir, split_ratio=0.8)