import React from 'react';
import '../styles/copyright.css'; // Make sure the path matches your project structure

const Copyright = () => {
  return (
    <footer className="copyright">
      <p>© {new Date().getFullYear()} Your Company Name. All rights reserved.</p>
    </footer>
  );
};

export default Copyright;
