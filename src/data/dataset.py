# src/data/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTDataModule:
    def __init__(self, data_dir: str = './data', batch_size: int = 32, use_augmentation: bool = True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Basic transform
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Training transform with augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(15),  # Random rotation
            transforms.RandomAffine(        # Random affine transformation
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]) if use_augmentation else self.test_transform
        
    def setup(self):
        """Prepare datasets"""
        self.train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        self.test_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )