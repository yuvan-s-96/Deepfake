import React from 'react';
import '../styles/copyright.css';

const Copyright = () => {
  return (
    <footer className="copyright">
      <p>© {new Date().getFullYear()} Deepfake Detection Technology. All rights reserved.</p>
    </footer>
  );
};

export default Copyright;