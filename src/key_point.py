import torch
import torch.nn as nn

class GestureClassifier(nn.Module):
    def __init__(self, num_keypoints=21, num_classes=6, dropout_rate=0.3, l2_reg=0.01):
        super(GestureClassifier, self).__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_keypoints * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, keypoints):
        """
        Входные данные: ключевые точки, извлечённые из MediaPipe.
        Размер входа: (batch_size, num_keypoints * 2) — для каждого ключевого положения (x, y).
        """
        return self.fc_block(keypoints)
