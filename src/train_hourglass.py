import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import FreiHAND_Dataset  # Датасет для FreiHAND
import configparser
from tqdm import tqdm
from hourglass_network import HourglassNetwork  # Импорт архитектуры Hourglass

# Чтение конфигурационного файла
config = configparser.ConfigParser()
config.read('config.ini')

# Параметры из конфигурационного файла
train_data_path = config['Paths']['freihand_train_images_path']
val_data_path = config['Paths']['freihand_val_images_path']
model_save_path = config['Paths']['hourglass_model_save_path']
train_annotations_path = config['Paths']['freihand_train_annotations']
num_keypoints = int(config['Model']['num_keypoints'])
image_size = tuple(map(int, config['Model']['image_size'].split(',')))
batch_size = int(config['Training']['batch_size'])
learning_rate = float(config['Training']['learning_rate'])
num_epochs = int(config['Training']['num_epochs'])
save_every = int(config['Training']['save_every'])

# Функция обучения одной эпохи
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, keypoints) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        keypoints = keypoints.to(device)

        optimizer.zero_grad()
        output = model(images)
        
        loss = criterion(output, keypoints)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader)
    return train_loss

# Функция валидации
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, keypoints) in enumerate(tqdm(dataloader, desc="Validating")):
            images = images.to(device)
            keypoints = keypoints.to(device)

            output = model(images)
            loss = criterion(output, keypoints)
            running_loss += loss.item()

    val_loss = running_loss / len(dataloader)
    return val_loss

# Основная функция обучения
def train_model():
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Архитектура Hourglass Network
    model = HourglassNetwork(num_keypoints=num_keypoints).to(device)

    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # MSE используется для предсказания координат ключевых точек

    # Трансформации изображений
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Загрузка данных из FreiHAND
    train_dataset = FreiHAND_Dataset(train_data_path, train_annotations_path, transform=transform)
    val_dataset = FreiHAND_Dataset(val_data_path, train_annotations_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Цикл обучения
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with validation loss {best_val_loss:.4f}")

        # Сохранение модели каждые несколько эпох
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    train_model()