# web/backend/mnist_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self, dropout_rate=0.3, num_filters=64):
        super().__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters*2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters*4)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters*4 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)