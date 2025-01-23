import torch
import logging
from pathlib import Path
from src.models.mnist_model import MNISTModel
from src.data.dataset import MNISTDataModule
from src.training.trainer import MNISTTrainer
from src.utils.visualization import MNISTVisualizer

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 80
    
    # Initialize data module
    logger.info("Loading data...")
    data_module = MNISTDataModule(
        data_dir='./data',
        batch_size=BATCH_SIZE,
        use_augmentation=True #artificially increase the size and diversity of a dataset 
    )
    data_module.setup()
    
    # Create data loaders
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Initialize model
    logger.info("Creating model...")
    model = MNISTModel(
        dropout_rate=0.3,
        num_filters=64
    )
    
    # Initialize trainer
    trainer = MNISTTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        test_loader=data_module.test_dataloader(),
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        use_scheduler=True
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train()
    
    # Save model
    save_dir = Path('models')
    save_dir.mkdir(exist_ok=True)
    trainer.save_model(save_dir / 'mnist_model.pth')
    
    # Visualize results
    logger.info("Creating visualizations...")
    visualizer = MNISTVisualizer()
    
    # Plot training history
    history_plot = visualizer.plot_training_history(history)
    history_plot.savefig('training_history.png')
    
    # Get sample predictions
    model.eval()
    test_images, test_labels = next(iter(test_loader))
    test_images, test_labels = test_images[:5].to(device), test_labels[:5]
    
    with torch.no_grad():
        predictions = model(test_images).argmax(dim=1)
    
    # Plot sample predictions
    samples_plot = visualizer.plot_samples(
        test_images,
        test_labels,
        predictions
    )
    samples_plot.savefig('sample_predictions.png')
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()