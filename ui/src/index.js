import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Check if we're running as an extension
const rootElement = document.getElementById('voicebridge-root') || document.getElementById('root');

if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
