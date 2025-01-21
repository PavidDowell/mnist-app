# tests/test_model.py
import pytest
import torch
import torch.nn as nn
from src.models.mnist_model import MNISTModel
from src.data.dataset import MNISTDataModule

class TestMNISTModel:
    @pytest.fixture
    def model(self):
        """Create a model instance for testing"""
        return MNISTModel()
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch of data"""
        return torch.randn(32, 1, 28, 28)  # batch_size=32, channels=1, height=28, width=28
    
    def test_model_structure(self, model):
        """Test if model has the correct structure"""
        # Check if model has expected layers
        assert hasattr(model, 'conv1'), "Model should have conv1 layer"
        assert hasattr(model, 'conv2'), "Model should have conv2 layer"
        assert hasattr(model, 'fc1'), "Model should have fc1 layer"
        assert hasattr(model, 'fc2'), "Model should have fc2 layer"
    
    def test_forward_pass(self, model, sample_batch):
        """Test if forward pass works and outputs correct shape"""
        output = model(sample_batch)
        
        # Check output shape (batch_size, num_classes)
        assert output.shape == (32, 10), f"Expected shape (32, 10), got {output.shape}"
        
        # Check if output is valid probability distribution
        assert torch.allclose(torch.exp(output).sum(dim=1), 
                            torch.tensor(1.0), 
                            atol=1e-6), "Output should sum to 1"
    
    def test_backward_pass(self, model, sample_batch):
        """Test if backward pass computes gradients correctly"""
        output = model(sample_batch)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients are computed
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient for {name} should not be None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
    
    def test_save_load(self, model, tmp_path):
        """Test if model can be saved and loaded correctly"""
        # Save model
        save_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        new_model = MNISTModel()
        new_model.load_state_dict(torch.load(save_path))
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.equal(param1, param2), f"Parameters {name1} don't match"

# tests/test_dataset.py
class TestMNISTDataModule:
    @pytest.fixture
    def data_module(self):
        """Create a data module instance for testing"""
        return MNISTDataModule(batch_size=32)
    
    def test_data_loading(self, data_module):
        """Test if data loads correctly"""
        data_module.setup()
        
        # Check if datasets are created
        assert data_module.train_dataset is not None, "Train dataset should exist"
        assert data_module.test_dataset is not None, "Test dataset should exist"
        
        # Check data loaders
        train_loader = data_module.train_dataloader()
        test_loader = data_module.test_dataloader()
        
        # Check batch shape
        x, y = next(iter(train_loader))
        assert x.shape[0] == 32, f"Expected batch size 32, got {x.shape[0]}"
        assert x.shape[1:] == (1, 28, 28), f"Expected shape (1, 28, 28), got {x.shape[1:]}"
        
        # Check labels
        assert torch.all((y >= 0) & (y <= 9)), "Labels should be between 0 and 9"
    
    def test_data_augmentation(self, data_module):
        """Test if data augmentation works"""
        data_module.setup()
        
        # Get same image twice
        dataset = data_module.train_dataset
        img1, _ = dataset[0]
        img2, _ = dataset[0]
        
        # With augmentation, they should be different
        if data_module.train_transform != data_module.test_transform:
            assert not torch.equal(img1, img2), "Augmented images should be different"