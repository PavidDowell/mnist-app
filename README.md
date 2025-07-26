# MNIST Digit Classifier

A deep learning application that classifies handwritten digits using the MNIST dataset. The project includes both a model training pipeline and a web interface for real-time digit classification. This is a very basic implementation and can be improved by adding more layers, data augmentation, and hyperparameter tuning.

## Project Structure

```plaintext
mnist_classifier/
├── src/
│   ├── data/         # Dataset handling
│   └── models/       # Model architecture
├── web/
│   ├── backend/      # Flask backend
│   └── frontend/     # Web interface
├── notebooks/        # Jupyter notebooks for experiments
└── requirements.txt  # Python dependencies
```

## Setup

1. Create and activate a virtual environment:
   
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

1. Train the model using the provided notebook in the notebooks/ directory:
   
   ```bash
   jupyter lab
   ```

2. Navigate to the training notebook and follow the experiment steps.

## Running the Web Application

1. Start the Flask backend:
   
   ```bash
   cd web/backend
   python app.py
   ```

2. In a new terminal window, start the React frontend:
   
   ```bash
   cd web/frontend
   npm install
   npm start
   ```

3. Open your browser and navigate to http://localhost:3000 to use the web application.

## Requirements

- Python 3.8+
- PyTorch
- Flask
- Node.js (for frontend)
- Additional dependencies listed in requirements.txt

## Model Architecture

The model uses a CNN architecture with:

- 3 convolutional blocks with batch normalization
- Dropout for regularization
- 2 fully connected layers
- Trained on the MNIST dataset