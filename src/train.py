import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import numpy as np
from key_point import GestureClassifier
from dataset import HandGestureDataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

# Функция для извлечения ключевых точек с помощью MediaPipe (предположим, что она уже определена в key_point.py)
from key_point import MediaPipeKeypointExtractor, get_keypoints

# Обучение одной эпохи
def train_epoch(model, dataloader, extractor, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, target_gestures) in enumerate(dataloader):
        images = [cv2.imread(img_path) for img_path in images]
        keypoints_batch = []

        for img in images:
            keypoints = get_keypoints(img, extractor)
            if keypoints is not None:
                keypoints_batch.append(keypoints)

        if len(keypoints_batch) > 0:
            keypoints_batch = torch.stack(keypoints_batch).to(device)
            target_gestures = target_gestures.to(device)

            optimizer.zero_grad()
            output = model(keypoints_batch)
            loss = criterion(output, target_gestures)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target_gestures.size(0)
            correct += (predicted == target_gestures).sum().item()

    train_loss = running_loss / len(dataloader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

# Валидация модели на валидационном наборе
def validate_epoch(model, dataloader, extractor, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch_idx, (images, target_gestures) in enumerate(dataloader):
            images = [cv2.imread(img_path) for img_path in images]
            keypoints_batch = []

            for img in images:
                keypoints = get_keypoints(img, extractor)
                if keypoints is not None:
                    keypoints_batch.append(keypoints)

            if len(keypoints_batch) > 0:
                keypoints_batch = torch.stack(keypoints_batch).to(device)
                target_gestures = target_gestures.to(device)

                output = model(keypoints_batch)
                loss = criterion(output, target_gestures)
                running_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total += target_gestures.size(0)
                correct += (predicted == target_gestures).sum().item()

                true_labels.extend(target_gestures.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc, true_labels, pred_labels

# Основная функция обучения с логированием и сохранением лучших весов
def train_model(num_epochs, train_loader, val_loader, model, optimizer, criterion, scheduler, device, trial=None):
    writer = SummaryWriter()
    extractor = MediaPipeKeypointExtractor()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, extractor, optimizer, criterion, device)
        val_loss, val_acc, true_labels, pred_labels = validate_epoch(model, val_loader, extractor, criterion, device)

        end_time = time.time()

        # Логирование
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Time per epoch', end_time - start_time, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')

        # Сохранение лучших весов
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gesture_classifier.pth')

        # Обновление learning rate
        scheduler.step()

    # Построение и сохранение матрицы ошибок
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")

    writer.close()

# Optuna для гиперпараметров
def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_keypoints = 21
    num_classes = 6
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    model = GestureClassifier(num_keypoints=num_keypoints, num_classes=num_classes, dropout_rate=dropout_rate).to(device)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # L2-регуляризация
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_dataset = HandGestureDataset('path/to/train', transform=None)
    val_dataset = HandGestureDataset('path/to/val', transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_model(20, train_loader, val_loader, model, optimizer, criterion, scheduler, device)

    # После обучения возвращаем лучшую валидационную потерю для Optuna
    return best_val_loss

# Запуск Optuna для подбора гиперпараметров
if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
