from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import sys
import os
from mnist_model import MNISTModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModel().to(device)

# Load the checkpoint and extract just the model state dict
checkpoint = torch.load("../../models/mnist_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Transform image
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get top 3 predictions
    top3_prob, top3_indices = torch.topk(probabilities[0], 3)
    predictions = [
        {
            "digit": idx.item(),
            "probability": prob.item()
        }
        for idx, prob in zip(top3_indices, top3_prob)
    ]
    
    return {
        "predictions": predictions,
        "success": True
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)