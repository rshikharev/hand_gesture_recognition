import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Базовый сверточный блок: Conv2D -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class HourglassBlock(nn.Module):
    """Hourglass блок для предсказания ключевых точек"""
    def __init__(self, num_features, depth=4):
        super(HourglassBlock, self).__init__()
        self.depth = depth
        self.features = nn.ModuleList([BasicBlock(num_features, num_features) for _ in range(depth)])
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling для downsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # Upsampling для обратного поднятия разрешения

    def forward(self, x):
        down_features = []
        # Downsampling блоки
        for i in range(self.depth):
            x = self.features[i](x)
            down_features.append(x)
            x = self.pool(x)

        # Upsampling блоки
        for i in reversed(range(self.depth)):
            x = self.upsample(x)
            x = x + down_features[i]  # Skip connection

        return x

class HourglassNetwork(nn.Module):
    """Hourglass Network для предсказания ключевых точек"""
    def __init__(self, num_keypoints, num_features=256, depth=4):
        super(HourglassNetwork, self).__init__()
        self.preprocess = nn.Sequential(
            BasicBlock(3, num_features),  # Предполагаем, что входное изображение RGB
            BasicBlock(num_features, num_features)
        )

        self.hourglass = HourglassBlock(num_features, depth=depth)
        self.out_layer = nn.Conv2d(num_features, num_keypoints, kernel_size=1)  # 1x1 Conv для получения предсказаний

    def forward(self, x):
        x = self.preprocess(x)
        x = self.hourglass(x)
        x = self.out_layer(x)  # Выход - heatmap ключевых точек
        return x
