# src/utils/visualization.py
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Tuple, Dict
import seaborn as sns

class MNISTVisualizer:
    def __init__(self):
        self.class_names = [str(i) for i in range(10)]
        
    def plot_samples(self, images: torch.Tensor, labels: torch.Tensor,
                    predictions: torch.Tensor = None, num_samples: int = 5):
        """Plot sample images with labels and predictions"""
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        
        for i, ax in enumerate(axes):
            if i < len(images):
                # Convert image to numpy and reshape
                img = images[i].cpu().numpy().squeeze()
                
                # Plot image
                ax.imshow(img, cmap='gray')
                
                # Set title with true label and prediction
                title = f'True: {labels[i]}'
                if predictions is not None:
                    title += f'\nPred: {predictions[i]}'
                ax.set_title(title)
                
                ax.axis('off')
                
        plt.tight_layout()
        return fig
        
    def plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(history['train_losses'], label='Train')
        ax1.plot(history['test_losses'], label='Test')
        ax1.set_title('Loss History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(history['train_accuracies'], label='Train')
        ax2.plot(history['test_accuracies'], label='Test')
        ax2.set_title('Accuracy History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        return fig
        
    def plot_confusion_matrix(self, true_labels: List[int],
                            predicted_labels: List[int]):
        """Plot confusion matrix"""
        cm = torch.zeros(10, 10, dtype=torch.int32)
        for t, p in zip(true_labels, predicted_labels):
            cm[t, p] += 1
            
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        return plt.gcf()