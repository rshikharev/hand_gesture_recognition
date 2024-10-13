import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        # Создаем папки для train и val
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        
        files = os.listdir(category_path)
        random.shuffle(files)
        
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        for file in train_files:
            shutil.move(os.path.join(category_path, file), os.path.join(train_dir, category, file))
        
        for file in val_files:
            shutil.move(os.path.join(category_path, file), os.path.join(val_dir, category, file))

source_dir = "data/raw"
train_dir = "data/train"
val_dir = "data/val"

split_data(source_dir, train_dir, val_dir)