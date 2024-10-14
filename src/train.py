import configparser
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from key_point import GestureClassifier, MediaPipeKeypointExtractor
from dataset import HandGestureDataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

# Получаем параметры из конфигурационного файла
train_data_path = config['Paths']['train_data_path']
val_data_path = config['Paths']['val_data_path']
model_save_path = config['Paths']['model_save_path']
confusion_matrix_save_path = config['Paths']['confusion_matrix_save_path']

num_keypoints = int(config['Model']['num_keypoints'])
num_classes = int(config['Model']['num_classes'])

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        Инициализация.
        :param patience: Сколько эпох ждать улучшений.
        :param delta: Минимальное изменение для квалификации как улучшение.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Функция для извлечения ключевых точек с помощью MediaPipe
def get_keypoints(image, extractor):
    keypoints = extractor.extract_keypoints(image)
    if keypoints is not None:
        return torch.tensor(keypoints, dtype=torch.float32)
    return None

# Обучение одной эпохи
def train_epoch(model, dataloader, extractor, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, target_gestures) in enumerate(dataloader):
        keypoints_batch = []
        valid_indices = []

        for i, img in enumerate(images):
            keypoints = extractor.extract_keypoints(img)
            if keypoints is not None:
                keypoints_batch.append(keypoints)
                valid_indices.append(i)

        # Пропускаем, если нет валидных данных
        if len(keypoints_batch) == 0:
            continue

        keypoints_batch = torch.tensor(keypoints_batch, dtype=torch.float32).to(device)
        target_gestures = target_gestures[valid_indices].to(device)

        optimizer.zero_grad()
        output = model(keypoints_batch)

        # Проверяем, что размеры батчей совпадают
        if output.size(0) != target_gestures.size(0):
            raise ValueError(f"Expected input batch_size ({output.size(0)}) to match target batch_size ({target_gestures.size(0)}).")

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
            keypoints_batch = []
            valid_indices = []

            for i, img in enumerate(images):
                keypoints = extractor.extract_keypoints(img)
                if keypoints is not None:
                    keypoints_batch.append(keypoints)
                    valid_indices.append(i)

            # Пропускаем, если нет валидных данных
            if len(keypoints_batch) == 0:
                continue

            keypoints_batch = torch.tensor(keypoints_batch, dtype=torch.float32).to(device)
            target_gestures = target_gestures[valid_indices].to(device)

            output = model(keypoints_batch)

            # Проверяем, что размеры батчей совпадают
            if output.size(0) != target_gestures.size(0):
                raise ValueError(f"Expected input batch_size ({output.size(0)}) to match target batch_size ({target_gestures.size(0)}).")

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
def train_model(num_epochs, train_loader, val_loader, model, optimizer, criterion, scheduler, device, patience=5):
    writer = SummaryWriter()
    extractor = MediaPipeKeypointExtractor()
    best_val_loss = float('inf')

    early_stopping = EarlyStopping(patience=patience)

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
            torch.save(model.state_dict(), model_save_path)

        # Ранняя остановка
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Обновление learning rate
        scheduler.step()

    # Построение и сохранение матрицы ошибок
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(confusion_matrix_save_path)

    writer.close()

# Запуск обучения с фиксированными гиперпараметрами
if __name__ == '__main__':
    print(f'torch.cuda.is_available() - {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Фиксированные гиперпараметры
    dropout_rate = 0.3
    learning_rate = 0.001
    batch_size = 64
    optimizer_name = "Adam"

    # Инициализация модели и оптимизатора
    model = GestureClassifier(num_keypoints=num_keypoints, num_classes=num_classes, dropout_rate=dropout_rate).to(device)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Загрузка данных
    train_dataset = HandGestureDataset(train_data_path, transform=None)
    val_dataset = HandGestureDataset(val_data_path, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Запуск обучения
    train_model(5, train_loader, val_loader, model, optimizer, criterion, scheduler, device, patience=5)