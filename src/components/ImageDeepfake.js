import React, { useState } from 'react';
import axios from 'axios';
import { Upload, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import Navbar from './NavBar';
import '../styles/imagedeepfake.css';
import '../styles/responsive-styles.css';

const Alert = ({ variant = 'info', title, children }) => {
  const bgColors = {
    success: 'bg-green-50 border-green-200',
    error: 'bg-red-50 border-red-200',
    info: 'bg-blue-50 border-blue-200'
  };

  return (
    <div className={`alert-container ${bgColors[variant]}`}>
      <div className="alert-content">
        <div className="alert-icon">
          {variant === 'error' && <AlertCircle className="icon error" />}
          {variant === 'success' && <CheckCircle className="icon success" />}
          {variant === 'info' && <AlertCircle className="icon info" />}
        </div>
        <div className="alert-text">
          <h3 className="alert-title">{title}</h3>
          <div className="alert-description">{children}</div>
        </div>
      </div>
    </div>
  );
};

const CustomSpinner = () => (
  <div className="custom-spinner-container">
    <Loader2 className="custom-spinner" />
  </div>
);

const DeepfakeImageDetector = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
    setResult(null);
    setError(null);
  };

  const detectDeepfake = async () => {
    if (!file) {
      setError('Please upload an image first');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5000/detect-image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResult(response.data);
      
    } catch (err) {
      setError(err.response?.data?.error || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="video-deepfake-wrapper">
      <Navbar />
      <div className="container video-deepfake-container">
        <div className="row">
          <div className="col">
            <div className="deepfake-image-detector">
              <div className="detector-header">
                <h1>AI Image Authenticity Detector</h1>
                <p>Upload an image to analyze its authenticity using advanced AI deepfake detection algorithms.</p>
              </div>

              <div className="video-upload-card">
                <div className="upload-form">
                  <div className="file-upload-section">
                    <label className="file-upload-label">
                      <Upload className="upload-icon" />
                      <span>Upload Image</span>
                      <input 
                        type="file" 
                        accept=".jpg,.jpeg,.png" 
                        onChange={handleFileChange}
                        className="file-input"
                      />
                    </label>
                    {file && (
                      <p className="selected-file-name">Selected: {file.name}</p>
                    )}
                  </div>

                  <button 
                    onClick={detectDeepfake} 
                    disabled={!file}
                    className="submit-button"
                  >
                    {loading ? 'Analyzing...' : 'Detect Deepfake'}
                  </button>
                </div>

                {loading && (
                  <div className="loading-container">
                    <CustomSpinner />
                    <p>Analyzing image...</p>
                  </div>
                )}

                {error && (
                  <Alert variant="error" title="Error">
                    {error}
                  </Alert>
                )}

                {result && (
                  <div className="result-container">
                    <Alert 
                      variant={result.prediction.toLowerCase() === 'real' ? 'success' : 'error'} 
                      title={`Detection Result: ${result.prediction}`}
                    >
                      {result.prediction.toLowerCase() === 'real' 
                        ? 'This image appears to be authentic.'
                        : 'This image appears to be AI-generated or manipulated.'}
                    </Alert>
                    
                    <div className="result-details">
                      <div className="confidence-breakdown">
                        <h3>Confidence Levels</h3>
                        <p>Real Image: {result.confidence.real.toFixed(2)}%</p>
                        <p>Fake Image: {result.confidence.fake.toFixed(2)}%</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeepfakeImageDetector;