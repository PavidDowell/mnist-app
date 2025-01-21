import React from 'react';

export const Predictions = ({ predictions }) => {
  if (!predictions) return null;

  return (
    <div className="predictions">
      <h3>Predictions:</h3>
      {predictions.map((pred, index) => (
        <div key={index} className="prediction-item">
          <span className="digit">Digit {pred.digit}</span>
          <span className="probability">
            {(pred.probability * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
};