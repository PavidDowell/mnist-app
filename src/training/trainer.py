# Training logic
# src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path

class MNISTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 0.001,
        epochs: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_scheduler: bool = False  # Added parameter
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize learning rate scheduler if requested
        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            )
        
        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics tracking
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(f'Train Batch [{batch_idx}/{len(self.train_loader)}] '
                               f'Loss: {loss.item():.6f}')
                
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
        
    def test_epoch(self):
        """Evaluate the model"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        # Calculate metrics
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(accuracy)
        
        # Update learning rate scheduler if it exists
        if self.scheduler is not None:
            self.scheduler.step(avg_loss)
        
        return avg_loss, accuracy
        
    def train(self):
        """Full training loop"""
        self.logger.info(f"Starting training on device: {self.device}")
        
        for epoch in range(self.epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Test epoch
            test_loss, test_acc = self.test_epoch()
            
            # Log progress
            self.logger.info(f"Epoch: {epoch+1}/{self.epochs}")
            self.logger.info(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.2f}%")
            
        return {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }
        
    def save_model(self, path: str):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")