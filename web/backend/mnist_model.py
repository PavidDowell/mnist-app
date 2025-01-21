# this is the model architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 128)  # 1600 = 64 * 5 * 5
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1600)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_features(self, x):
        """Extract features for visualization"""
        features = []
        
        # First conv layer features
        x = self.conv1(x)
        features.append(('conv1', x))
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        # Second conv layer features
        x = self.conv2(x)
        features.append(('conv2', x))
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        return features