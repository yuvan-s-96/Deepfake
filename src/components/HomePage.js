import React from 'react';
import Navbar from './NavBar';
import Copyright from './CopyRight';
import Carousel from './Carousel';
import '../styles/Homepage.css';
import '../styles/responsive-styles.css';  // New responsive global styles

const HomePage = () => {
  return (
    <div className="homepage-wrapper">
      <Navbar />
      <div className="container homepage-content">
        <div className="row">
          <div className="col">
            <h1 className="homepage-title">Welcome to Deepfake Detection</h1>
            <p className="homepage-description">
              Explore advanced techniques and tools for detecting deepfake content across various media formats.
            </p>
          </div>
        </div>
        <div className="row">
          <div className="col">
            <Carousel />
          </div>
        </div>
        <div className="row">
          <div className="col">
            <Copyright />
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;