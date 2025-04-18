import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './components/HomePage';
import AboutPage from './components/AboutPage'; 
import VideoDeepfake from './components/VideoDeepfake';
import AudioDeepfake from './components/AudioDeepfake';
import ImageDeepfake from './components/ImageDeepfake';

const App = () => {
  return (
    <Router>
      <div>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/features/video-deepfake" element={<VideoDeepfake />} />
          <Route path="/features/audio-deepfake" element={<AudioDeepfake />} />
          <Route path="/features/image-deepfake" element={<ImageDeepfake />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
