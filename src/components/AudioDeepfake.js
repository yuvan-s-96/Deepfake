import React, { useState } from 'react';
import Navbar from './NavBar';
import '../styles/Featurepage.css';

const AudioDeepfake = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    
    // Validate file type and size
    const allowedTypes = [
      'audio/wav', 
      'audio/mp3', 
      'audio/mpeg', 
      'audio/flac', 
      'audio/x-flac'
    ];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!allowedTypes.includes(selectedFile.type)) {
      setError('Invalid file type. Please upload WAV, MP3, or FLAC file.');
      return;
    }

    if (selectedFile.size > maxSize) {
      setError('File is too large. Maximum file size is 10MB.');
      return;
    }

    setFile(selectedFile);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!file) {
      setError('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('original_filename', file.name);

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/detect-audio', {
        method: 'POST',
        body: formData
      });

      const responseText = await response.text();
      console.log('Raw Response:', responseText);

      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('JSON Parsing Error:', parseError);
        console.error('Response Text:', responseText);
        throw new Error('Failed to parse server response');
      }

      if (!response.ok) {
        throw new Error(data.error || 'Network response was not ok');
      }

      console.log('Parsed Response Data:', data);

      // Log raw API response details
      if (data.output === undefined) {
        console.error('Unexpected response structure:', data);
        throw new Error('Unexpected response from server');
      }

      setResult(data);
    } catch (error) {
      console.error('Detailed Error:', error);
      setError(`An error occurred: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Navbar />
      <div className="feature-content">
        <h1>Audio Deepfake Detection</h1>
        <p>Upload a WAV, MP3, or FLAC audio file (max 10MB) to check for deepfakes.</p>
        
        <form onSubmit={handleSubmit}>
          <input 
            type="file" 
            onChange={handleFileChange} 
            accept=".wav,.mp3,.flac,audio/wav,audio/mp3,audio/mpeg,audio/flac,audio/x-flac" 
          />
          <button 
            type="submit" 
            disabled={loading || !file}
          >
            {loading ? 'Detecting...' : 'Upload'}
          </button>
        </form>
        
        {error && (
          <div className="error-box">
            <p style={{ color: 'red' }}>{error}</p>
          </div>
        )}
        
        {result && (
          <div className="detailed-results">
            <div className={`result-box ${result.output === 'FAKE' ? 'fake-result' : 'real-result'}`}>
              <h2>Detection Result:</h2>
              <p>Audio is {result.output || 'INCONCLUSIVE'}</p>
              <p>Overall Confidence: {result.confidence ? result.confidence.toFixed(2) : 'N/A'}%</p>
              
              {/* New Percentages Section */}
              {result.percentages && (
                <div className="percentages-breakdown">
                  <h3>Detailed Breakdown</h3>
                  <p>Fake Probability: {result.percentages.fake.toFixed(2)}%</p>
                  <p>Real Probability: {result.percentages.real.toFixed(2)}%</p>
                </div>
              )}
            </div>

            {result.spectrogram && (
              <div className="spectrogram-section">
                <h2>Frequency Analysis</h2>
                <img 
                  src={`data:image/png;base64,${result.spectrogram}`} 
                  alt="Audio Spectrogram" 
                  className="spectrogram-image"
                />
              </div>
            )}

            {result.spectral_features && (
              <div className="spectral-features">
                <h3>Spectral Features</h3>
                <table>
                  <tbody>
                    {Object.entries(result.spectral_features).map(([key, value]) => (
                      <tr key={key}>
                        <td>{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</td>
                        <td>{typeof value === 'number' ? value.toFixed(2) : value} Hz</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioDeepfake;