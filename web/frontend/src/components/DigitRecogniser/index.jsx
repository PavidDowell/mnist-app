import React, { useState } from 'react';
import axios from 'axios';
import { Canvas } from './Canvas';
import { Predictions } from './Predictions';

const DigitRecogniser = () => {
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleDrawingComplete = async (canvas) => {
    try {
      setIsLoading(true);
      setError(null);

      // Convert canvas to blob
      const blob = await new Promise(resolve => 
        canvas.toBlob(resolve, 'image/png')
      );

      // Create form data
      const formData = new FormData();
      formData.append('file', blob);

      // Send to backend
      const response = await axios.post(
        'http://localhost:8000/predict',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      setPredictions(response.data.predictions);
    } catch (err) {
      setError('Failed to get prediction');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="digit-recognizer">
      <h1>Digit Recogniser</h1>
      <Canvas onDrawingComplete={handleDrawingComplete} />
      {isLoading && <div className="loading">Processing...</div>}
      {error && <div className="error">{error}</div>}
      <Predictions predictions={predictions} />
    </div>
  );
};

export default DigitRecogniser;