import React, { useState } from 'react';
import axios from 'axios';
import { Upload, AlertCircle, CheckCircle, Loader2, Info } from 'lucide-react';
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

const XAIVisualizationCard = ({ title, imageData }) => {
  if (!imageData) return null;
  
  return (
    <div className="xai-visualization-card">
      <h4>{title}</h4>
      <div className="xai-image-container">
        <img src={`data:image/png;base64,${imageData}`} alt={title} />
      </div>
    </div>
  );
};

const DeepfakeImageDetector = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [showXAIExplanation, setShowXAIExplanation] = useState(false);
  

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setResult(null);
      setError(null);
      
      // Create preview URL for selected image
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(uploadedFile);
    }
  };

  const detectDeepfake = async () => {
    if (!file) {
      setError('Please upload an image first');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('generate_xai', 'true'); // Request XAI visualizations

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

  const toggleXAIExplanation = () => {
    setShowXAIExplanation(!showXAIExplanation);
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

                  {previewUrl && (
                    <div className="image-preview-container">
                      <h3>Image Preview</h3>
                      <img src={previewUrl} alt="Preview" className="image-preview" />
                    </div>
                  )}

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
                        <div className="confidence-bars">
                          <div className="confidence-bar-container">
                            <span>Real: {result.confidence.real.toFixed(2)}%</span>
                            <div className="confidence-bar">
                              <div className="confidence-bar-fill real" style={{ width: `${result.confidence.real}%` }}></div>
                            </div>
                          </div>
                          <div className="confidence-bar-container">
                            <span>Fake: {result.confidence.fake.toFixed(2)}%</span>
                            <div className="confidence-bar">
                              <div className="confidence-bar-fill fake" style={{ width: `${result.confidence.fake}%` }}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Display pixel distribution histogram */}
                      {result.visualizations?.histogram && (
                        <div className="image-analysis-section">
                          <h3>Pixel Distribution Analysis</h3>
                          <div className="visualization-image">
                            <img src={`data:image/png;base64,${result.visualizations.histogram}`} alt="Pixel Distribution Histogram" />
                          </div>
                        </div>
                      )}
                      
                      {/* XAI Visualizations */}
                      {result.xai_visualizations && (
                        <div className="xai-section">
                          <div className="xai-header" onClick={toggleXAIExplanation}>
                            <h3>Explainable AI Analysis <Info className="info-icon" size={18} /></h3>
                            <button className="toggle-xai-button">
                              {showXAIExplanation ? 'Hide Details' : 'Show Details'}
                            </button>
                          </div>
                          
                          {showXAIExplanation && (
                            <div className="xai-explanation">
                              <p>These visualizations highlight the areas of the image that most influenced the AI's decision:</p>
                              <ul>
                                <li><strong>Integrated Gradients:</strong> Shows which pixels were most important for the classification.</li>
                                <li><strong>Grad-CAM:</strong> Highlights regions the AI focused on when making its decision.</li>
                                <li><strong>Occlusion Analysis:</strong> Shows how prediction changes when different parts are blocked.</li>
                              </ul>
                            </div>
                          )}
                          
                          <div className="xai-visualizations-grid">
                            <XAIVisualizationCard 
                              title="Integrated Gradients" 
                              imageData={result.xai_visualizations.integrated_gradients} 
                            />
                            <XAIVisualizationCard 
                              title="Grad-CAM" 
                              imageData={result.xai_visualizations.grad_cam} 
                            />
                            <XAIVisualizationCard 
                              title="Occlusion Analysis" 
                              imageData={result.xai_visualizations.occlusion} 
                            />
                          </div>
                        </div>
                      )}
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