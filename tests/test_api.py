# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
from web.backend.main import app

class TestAPI:
    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a white image with a black digit
        img = Image.new('L', (28, 28), 'white')
        pixels = img.load()
        # Draw a simple vertical line (like digit 1)
        for y in range(5, 23):
            pixels[14, y] = 0
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_predict_endpoint(self, client, sample_image):
        """Test prediction endpoint"""
        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_image, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "predictions" in data
        assert "success" in data
        assert data["success"] == True
        
        # Check predictions
        predictions = data["predictions"]
        assert len(predictions) == 3  # Top 3 predictions
        
        for pred in predictions:
            assert "digit" in pred
            assert "probability" in pred
            assert 0 <= pred["digit"] <= 9
            assert 0 <= pred["probability"] <= 1
    
    def test_invalid_image(self, client):
        """Test prediction with invalid image"""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code != 200  # Should not accept invalid image
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, sample_image):
        """Test handling multiple concurrent requests"""
        import asyncio
        
        async def make_request():
            response = client.post(
                "/predict",
                files={"file": ("test.png", sample_image, "image/png")}
            )
            return response.status_code
        
        # Make 5 concurrent requests
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All requests should be successful
        assert all(status == 200 for status in results)

# tests/test_training.py
class TestTraining:
    @pytest.fixture
    def trainer(self):
        """Create a trainer instance for testing"""
        model = MNISTModel()
        data_module = MNISTDataModule(batch_size=32)
        data_module.setup()
        
        return trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            test_loader=data_module.test_dataloader(),
            epochs=1
        )
    
    def test_single_epoch(self, trainer):
        """Test if training for one epoch works"""
        history = trainer.train()
        
        # Check if history contains expected metrics
        assert "train_losses" in history
        assert "test_losses" in history
        assert "train_accuracies" in history
        assert "test_accuracies" in history
        
        # Check if metrics are reasonable
        assert all(0 <= acc <= 100 for acc in history["train_accuracies"])
        assert all(loss >= 0 for loss in history["train_losses"])
    
    def test_model_improvement(self, trainer):
        """Test if model improves during training"""
        history = trainer.train()
        
        # Check if loss decreases
        train_losses = history["train_losses"]
        assert train_losses[0] > train_losses[-1], "Loss should decrease during training"