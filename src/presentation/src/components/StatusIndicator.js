import React from 'react';
import './StatusIndicator.css';

const StatusIndicator = ({ status, error, isLightMode }) => {
  const getStatusInfo = () => {
    switch (status) {
      case 'recording':
        return { text: 'Recording...', color: '#ffffff', icon: '⏺️' };
      case 'listening':
        return { text: 'Listening...', color: '#ffffff', icon: '🎤' };
      case 'processing':
        return { text: 'Processing...', color: '#cccccc', icon: '⚙️' };
      case 'confirmed':
        return { text: 'Command Ready', color: '#ffffff', icon: '✓' };
      case 'executed':
        return { text: 'Command Executed', color: '#ffffff', icon: '✓' };
      case 'idle':
      default:
        return { text: 'Ready', color: '#808080', icon: '○' };
    }
  };

  const statusInfo = getStatusInfo();

  if (error) {
    return (
      <div className={`status-indicator error ${isLightMode ? 'light-mode' : 'dark-mode'}`}>
        <span className="status-icon">⚠️</span>
        <span className="status-text">{error}</span>
      </div>
    );
  }

  return (
    <div className={`status-indicator ${isLightMode ? 'light-mode' : 'dark-mode'}`} style={{ '--status-color': statusInfo.color }}>
      <span className="status-icon">{statusInfo.icon}</span>
      <span className="status-text">{statusInfo.text}</span>
    </div>
  );
};

export default StatusIndicator;
