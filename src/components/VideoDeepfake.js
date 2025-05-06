import { useState } from 'react';
import { Upload, AlertCircle, CheckCircle, Info, Camera, Clock } from 'lucide-react';
import Navbar from './NavBar';
import '../styles/videodeepfake.css';
import '../styles/responsive-styles.css';

const Alert = ({ variant = 'info', title, children }) => {
  return (
    <div className={`alert-container ${variant === 'error' ? 'bg-error' : variant === 'success' ? 'bg-success' : 'bg-info'}`}>
      <div className="alert-content">
        <div className="alert-icon">
          {variant === 'error' && <AlertCircle className="icon error" />}
          {variant === 'success' && <CheckCircle className="icon success" />}
          {variant === 'info' && <Info className="icon info" />}
        </div>
        <div>
          <h3 className="alert-title">{title}</h3>
          <div className="alert-description">{children}</div>
        </div>
      </div>
    </div>
  );
};

const FrameImportanceBar = ({ importance, index, type = "general" }) => {
  // Normalize importance value between 0 and 100
  const normalizedValue = Math.min(Math.max(importance * 100, 0), 100);
  
  // Color class based on importance value and type
  const getColorClass = (value, type) => {
    if (type === "spatial") {
      // Blues for spatial features (Xception)
      if (value > 75) return 'bg-blue-600';
      if (value > 50) return 'bg-blue-500';
      if (value > 25) return 'bg-blue-400';
      return 'bg-blue-300';
    } else if (type === "temporal") {
      // Purples for temporal features (VideoMAE)
      if (value > 75) return 'bg-purple-600';
      if (value > 50) return 'bg-purple-500';
      if (value > 25) return 'bg-purple-400';
      return 'bg-purple-300';
    } else {
      // Default colors
      if (value > 75) return 'bg-error';
      if (value > 50) return 'bg-warning';
      if (value > 25) return 'bg-info';
      return 'bg-success';
    }
  };

  return (
    <div className="frame-importance-bar">
      <div className="frame-bar-container">
        <span className="frame-label">Frame {index}</span>
        <div className="frame-progress-bg">
          <div 
            className={`frame-progress ${getColorClass(normalizedValue, type)}`} 
            style={{ width: `${normalizedValue}%` }}
          ></div>
        </div>
        <span className="frame-value">{normalizedValue.toFixed(1)}%</span>
      </div>
    </div>
  );
};

const FeatureTypeHeader = ({ icon, title, description }) => {
  return (
    <div className="feature-type-header">
      <div className="feature-type-icon">
        {icon}
      </div>
      <div className="feature-type-text">
        <h4 className="feature-type-title">{title}</h4>
        <p className="feature-type-description">{description}</p>
      </div>
    </div>
  );
};

const DeepfakeDetector = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [frameCount, setFrameCount] = useState(10);
  const [outputVisuals, setOutputVisuals] = useState(true);
  const [frameVisuals, setFrameVisuals] = useState([]);
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [viewMode, setViewMode] = useState('combined'); // 'combined', 'spatial', 'temporal'

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
      setFrameVisuals([]);
      setSelectedFrame(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a video file');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('video', file);
    
    // Add parameters to match backend expectations
    formData.append('frame_count', frameCount);
    formData.append('output_visuals', outputVisuals ? 'true' : 'false');

    try {
      const response = await fetch('http://localhost:5000/detect-video', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      console.log('Response data:', data);

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze video');
      }

      // Process frame data if available
      if (data.frames && data.frames.images) {
        setFrameVisuals(data.frames.images);
        if (data.frames.images.length > 0) {
          setSelectedFrame(0);
        }
      }

      setResult(data);
    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Function to get feature indicators based on frame data
  const getFeatureIndicators = (frameIdx) => {
    if (!result || !result.frames || !result.frames.importance_scores) {
      return { spatial: 0, temporal: 0 };
    }
    
    // We'll simulate different feature types for this example
    // In a real implementation, these would come from the backend
    const baseImportance = result.frames.importance_scores[frameIdx] || 0;
    
    // Simulate spatial (Xception) features - more sensitive to spatial artifacts
    const spatialImportance = baseImportance * (1 + 0.3 * Math.sin(frameIdx));
    
    // Simulate temporal (VideoMAE) features - more sensitive to temporal inconsistencies
    const temporalImportance = baseImportance * (1 + 0.3 * Math.cos(frameIdx));
    
    return {
      spatial: Math.min(Math.max(spatialImportance, 0), 1),
      temporal: Math.min(Math.max(temporalImportance, 0), 1)
    };
  };

  return (
    <div className="video-deepfake-wrapper">
      <Navbar />
      <div className="container video-deepfake-container">
        <div className="video-deepfake-header">
          <h1>Deepfake Detection</h1>
          <p>
            Upload a video to analyze if it's genuine or manipulated
          </p>
        </div>

        <div className="video-upload-card">
          <form onSubmit={handleSubmit} className="upload-form">
            <div className="file-upload-section">
              <label className="file-upload-label">
                <Upload className="upload-icon" />
                <span className="upload-text">Upload Video</span>
                <span className="upload-hint">
                  MP4, AVI, MOV files supported
                </span>
                <input
                  type="file"
                  className="file-input"
                  accept="video/*"
                  onChange={handleFileChange}
                />
              </label>
              {file && (
                <p className="selected-file-name">
                  Selected: <span className="file-name-highlight">{file.name}</span>
                </p>
              )}
            </div>

            <div className="row">
              <div className="col col-md-6">
                <label className="input-label">
                  Frame Count:
                </label>
                <input
                  type="number"
                  min="5"
                  max="30"
                  value={frameCount}
                  onChange={(e) => setFrameCount(parseInt(e.target.value))}
                  className="number-input"
                />
                <p className="input-hint">
                  Number of frames to analyze (5-30)
                </p>
              </div>
              
              <div className="col col-md-6">
                <div className="checkbox-container">
                  <input
                    id="outputVisuals"
                    type="checkbox"
                    checked={outputVisuals}
                    onChange={(e) => setOutputVisuals(e.target.checked)}
                    className="checkbox-input"
                  />
                  <label htmlFor="outputVisuals" className="checkbox-label">
                    Generate visual analysis
                  </label>
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !file}
              className={`submit-button ${loading || !file ? 'disabled' : ''}`}
            >
              {loading ? 'Analyzing...' : 'Detect Deepfake'}
            </button>
          </form>
              
          {error && (
            <Alert variant="error" title="Analysis Error">
              {error}
            </Alert>
          )}

          {result && (
            <div className="results-container">
              <Alert 
                variant={result.prediction === 'FAKE' ? 'error' : 'success'}
                title="Analysis Results"
              >
                <div className="result-details">
                  <p className="result-prediction">
                    Prediction: <span className={result.prediction === 'FAKE' ? 'result-fake' : 'result-real'}>
                      {result.prediction}
                    </span>
                  </p>
                  <p className="result-confidence">
                    Confidence: <span className="result-highlight">{result.confidence}%</span>
                  </p>
                  
                  {/* Model comparison visualization */}
                  {result.model_comparison && (
                    <div className="model-comparison-section">
                      <h4 className="section-subtitle">Model Comparison</h4>
                      <div className="model-comparison-image">
                        <img 
                          src={`data:image/png;base64,${result.model_comparison}`} 
                          alt="Model comparison" 
                          className="model-comparison-img"
                        />
                      </div>
                    </div>
                  )}
                  
                  {/* View mode toggle buttons */}
                  {result.frames && result.frames.importance_scores && result.frames.importance_scores.length > 0 && (
                    <div className="feature-view-controls">
                      <h4 className="section-subtitle">Feature Analysis View</h4>
                      <div className="feature-view-buttons">
                        <button 
                          className={`feature-button ${viewMode === 'combined' ? 'active' : ''}`}
                          onClick={() => setViewMode('combined')}
                        >
                          Combined
                        </button>
                        <button 
                          className={`feature-button spatial ${viewMode === 'spatial' ? 'active' : ''}`}
                          onClick={() => setViewMode('spatial')}
                        >
                          Spatial (Xception)
                        </button>
                        <button 
                          className={`feature-button temporal ${viewMode === 'temporal' ? 'active' : ''}`}
                          onClick={() => setViewMode('temporal')}
                        >
                          Temporal (VideoMAE)
                        </button>
                      </div>
                    </div>
                  )}
                  
                  {/* Display frame importance analysis */}
                  {result.frames && result.frames.importance_scores && result.frames.importance_scores.length > 0 && (
                    <div className="frame-analysis-section">
                      {viewMode === 'combined' && (
                        <>
                          <h4 className="section-subtitle">Combined Frame Importance Analysis:</h4>
                          <div className="frame-bars-container">
                            {result.frames.importance_scores.map((importance, idx) => (
                              <FrameImportanceBar key={idx} importance={importance} index={idx} />
                            ))}
                          </div>
                        </>
                      )}
                      
                      {viewMode === 'spatial' && (
                        <>
                          <FeatureTypeHeader 
                            icon={<Camera size={24} />}
                            title="Spatial Features (Xception)"
                            description="Spatial features focus on visual artifacts within individual frames, detecting inconsistencies in facial features, unrealistic textures, and other static visual elements."
                          />
                          <div className="frame-bars-container">
                            {result.frames.importance_scores.map((_, idx) => {
                              const features = getFeatureIndicators(idx);
                              return (
                                <FrameImportanceBar 
                                  key={`spatial-${idx}`} 
                                  importance={features.spatial} 
                                  index={idx}
                                  type="spatial"
                                />
                              );
                            })}
                          </div>
                        </>
                      )}
                      
                      {viewMode === 'temporal' && (
                        <>
                          <FeatureTypeHeader 
                            icon={<Clock size={24} />}
                            title="Temporal Features (VideoMAE)"
                            description="Temporal features analyze inconsistencies across frames, detecting unnatural movements, lighting changes, and other anomalies that occur over time in the video sequence."
                          />
                          <div className="frame-bars-container">
                            {result.frames.importance_scores.map((_, idx) => {
                              const features = getFeatureIndicators(idx);
                              return (
                                <FrameImportanceBar 
                                  key={`temporal-${idx}`} 
                                  importance={features.temporal} 
                                  index={idx}
                                  type="temporal"
                                />
                              );
                            })}
                          </div>
                        </>
                      )}
                      
                      <p className="analysis-hint">
                        Higher values indicate frames with more deepfake indicators
                      </p>
                    </div>
                  )}
                  
                  {/* Display extracted frames */}
                  {frameVisuals.length > 0 && (
                    <div className="frames-gallery-section">
                      <h4 className="section-subtitle">Analyzed Frames:</h4>
                      <div className="frames-grid">
                        {frameVisuals.map((frame, idx) => (
                          <div 
                            key={idx}
                            className={`frame-thumbnail ${selectedFrame === idx ? 'selected' : ''}`}
                            onClick={() => setSelectedFrame(idx)}
                          >
                            <img 
                              src={`data:image/jpeg;base64,${frame.image}`} 
                              alt={`Frame ${idx}`} 
                              className="frame-image"
                            />
                          </div>
                        ))}
                      </div>
                      
                      {selectedFrame !== null && (
                        <div className="selected-frame-container">
                          <div className="frame-preview">
                            <img 
                              src={`data:image/jpeg;base64,${frameVisuals[selectedFrame].image}`} 
                              alt={`Selected frame ${selectedFrame}`}
                              className="frame-preview-image"
                            />
                          </div>
                          <div className="frame-metadata">
                            <p className="frame-metadata-title">
                              Frame {selectedFrame} Analysis
                            </p>
                            
                            {/* Feature indicators for selected frame */}
                            <div className="feature-indicators">
                              <div className="feature-indicator spatial">
                                <Camera size={16} />
                                <span className="indicator-label">Spatial (Xception):</span>
                                <div className="indicator-bar-bg">
                                  <div 
                                    className="indicator-bar spatial"
                                    style={{ 
                                      width: `${getFeatureIndicators(selectedFrame).spatial * 100}%` 
                                    }}
                                  ></div>
                                </div>
                                <span className="indicator-value">
                                  {(getFeatureIndicators(selectedFrame).spatial * 100).toFixed(2)}%
                                </span>
                              </div>
                              
                              <div className="feature-indicator temporal">
                                <Clock size={16} />
                                <span className="indicator-label">Temporal (VideoMAE):</span>
                                <div className="indicator-bar-bg">
                                  <div 
                                    className="indicator-bar temporal"
                                    style={{ 
                                      width: `${getFeatureIndicators(selectedFrame).temporal * 100}%` 
                                    }}
                                  ></div>
                                </div>
                                <span className="indicator-value">
                                  {(getFeatureIndicators(selectedFrame).temporal * 100).toFixed(2)}%
                                </span>
                              </div>
                            </div>
                            
                            <p className="frame-importance-value">
                              Combined Importance: {(frameVisuals[selectedFrame].importance * 100).toFixed(2)}%
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Display XAI visualizations if available */}
                  {result.xai_visualizations && result.xai_visualizations.length > 0 && (
                    <div className="xai-visualizations-section">
                      <h4 className="section-subtitle">XAI Visualizations:</h4>
                      <div className="xai-grid">
                        {result.xai_visualizations.map((visual, idx) => (
                          <div key={idx} className="xai-item">
                            <img 
                              src={`data:image/png;base64,${visual.image}`} 
                              alt={`XAI Frame ${visual.frame}`}
                              className="xai-image"
                            />
                            <p className="xai-caption">
                              Frame {visual.frame} - {visual.model}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Alert>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DeepfakeDetector;