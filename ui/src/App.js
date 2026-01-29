import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import SpeechInput from './components/SpeechInput';
import StatusIndicator from './components/StatusIndicator';
import FeedbackDisplay from './components/FeedbackDisplay';
import ErrorFeedback from './utils/ErrorFeedback';
import UiClient from './utils/UiClient';
import AuditLogger from './utils/auditLogger';
import AccessibilityLayer from './utils/accessibilityLayer';

// Set this to your backend endpoint, e.g. http://localhost:8000/audio
const AUDIO_BACKEND_URL = 'http://localhost:8000/audio';

function App() {
  const [isListening, setIsListening] = useState(true);
  const [status, setStatus] = useState('listening'); // idle, listening
  const [error, setError] = useState(null);
  const [isLightMode, setIsLightMode] = useState(false);
  const [feedbackItems, setFeedbackItems] = useState([]);

  // Initialize ErrorFeedback system
  const errorFeedbackRef = useRef(null);
  const uiClientRef = useRef(null);
  const auditLoggerRef = useRef(null);
  const accessibilityLayerRef = useRef(null);

  useEffect(() => {
    // Initialize modules
    auditLoggerRef.current = new AuditLogger();
    accessibilityLayerRef.current = new AccessibilityLayer();
    uiClientRef.current = new UiClient(setFeedbackItems, setError);
    errorFeedbackRef.current = new ErrorFeedback(
      uiClientRef.current,
      auditLoggerRef.current,
      accessibilityLayerRef.current
    );

    return () => {
      // Cleanup
      if (errorFeedbackRef.current) {
        errorFeedbackRef.current.clearAll();
      }
    };
  }, []);

  // Keyboard: light mode toggle (Alt+L) and collapse (Escape)
  useEffect(() => {
    const handleKeyDown = (e) => {
      const root = document.getElementById('voicebridge-root');
      const isWidgetVisible = root && root.style.display !== 'none';

      if (!isWidgetVisible) return;

      // Collapse widget: Escape
      if (e.key === 'Escape') {
        root.style.display = 'none';
        e.preventDefault();
        return;
      }
      // Toggle light mode: Alt+L (L for light)
      if (e.altKey && (e.key === 'l' || e.key === 'L')) {
        setIsLightMode(prev => !prev);
        e.preventDefault();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleAudioData = async (blob) => {
    // Send each 10s chunk to the backend server.
    setStatus('listening');

    if (!AUDIO_BACKEND_URL) return;

    try {
      const formData = new FormData();
      const mimeType = blob.type || 'audio/webm';
      const timestamp = Date.now();

      formData.append('audio', blob, `chunk-${timestamp}.webm`);
      formData.append('mimeType', mimeType);
      formData.append('timestamp', String(timestamp));

      const response = await fetch(AUDIO_BACKEND_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
    } catch (e) {
      console.error('Error sending audio chunk to backend:', e);
      
      // Use ErrorFeedback to show error (only if not already shown)
      if (errorFeedbackRef.current) {
        const errorCode = e.message.includes('Failed to fetch') || e.message.includes('NetworkError')
          ? 'NETWORK_ERROR'
          : 'BACKEND_ERROR';
        
        const errorDetail = e.message || 'Failed to send audio to server.';
        const feedbackId = errorFeedbackRef.current.showError(errorCode, errorDetail);
        
        // Only stop listening if this is a new error (not duplicate)
        if (feedbackId) {
          setStatus('idle');
          setIsListening(false);
        }
      } else {
        // Fallback to old error handling
        setError('Failed to send audio to server.');
        setStatus('idle');
        setIsListening(false);
      }
    }
  };

  const handleError = (errorMessage) => {
    // Map common error messages to error codes
    let errorCode = 'GENERIC_ERROR';
    if (errorMessage.includes('microphone') || errorMessage.includes('Microphone')) {
      errorCode = 'MIC_PERMISSION_DENIED';
    } else if (errorMessage.includes('device') || errorMessage.includes('in use')) {
      errorCode = 'AUDIO_DEVICE_ERROR';
    }

    // Use ErrorFeedback to show error
    if (errorFeedbackRef.current) {
      errorFeedbackRef.current.showError(errorCode, errorMessage);
    } else {
      // Fallback to old error handling
      setError(errorMessage);
    }
    
    setStatus('idle');
    setIsListening(false);
  };

  const handleDismissFeedback = (feedbackId) => {
    if (errorFeedbackRef.current) {
      errorFeedbackRef.current.dismiss(feedbackId);
    }
  };

  const handleRecoverySelect = (feedbackId, optionId) => {
    const item = feedbackItems.find(f => f.id === feedbackId);
    if (!item || !item.recoveryOptions) return;

    const option = item.recoveryOptions.options.find(o => o.id === optionId);
    if (!option) return;

    // Handle recovery actions
    console.log('Recovery option selected:', option.label);
    
    // Example recovery actions
    if (option.label.includes('Retry')) {
      // Retry the failed operation
      setIsListening(true);
      setStatus('listening');
    } else if (option.label.includes('Reload')) {
      window.location.reload();
    } else if (option.label.includes('Settings')) {
      // Open settings (could navigate to chrome://settings/content/microphone)
      window.open('chrome://settings/content/microphone', '_blank');
    }

    // Dismiss the feedback after action
    handleDismissFeedback(feedbackId);
  };

  const toggleTheme = () => {
    setIsLightMode(!isLightMode);
  };

  const handleCollapse = () => {
    const root = document.getElementById('voicebridge-root');
    if (root) root.style.display = 'none';
  };

  return (
    <div
      className={`App extension-overlay ${isLightMode ? 'light-mode' : 'dark-mode'}`}
      role="application"
      aria-label="VoiceBridge"
    >
      <div className="extension-container" tabIndex={0}>
        <header className="App-header">
          <div className="header-content">
            <div>
              <h1>VoiceBridge</h1>
              <p className="subtitle">Continuous audio capture</p>
            </div>
            <div className="header-controls">
              <button
                type="button"
                className="theme-toggle"
                onClick={toggleTheme}
                aria-label={isLightMode ? 'Switch to dark mode (Alt+L)' : 'Switch to light mode (Alt+L)'}
                title={isLightMode ? 'Dark mode (Alt+L)' : 'Light mode (Alt+L)'}
              >
                {isLightMode ? '🌙' : '☀️'}
              </button>
              <button
                type="button"
                className="minimize-button"
                onClick={handleCollapse}
                aria-label="Collapse widget (Escape)"
                title="Collapse (Escape)"
              >
                −
              </button>
            </div>
          </div>
        </header>

        <main className="App-main">
          <div className="main-container">
            <div className="left-panel">
              <StatusIndicator status={status} error={error} isLightMode={isLightMode} />
              <SpeechInput
                isListening={isListening}
                onAudioData={handleAudioData}
                onError={handleError}
                isLightMode={isLightMode}
              />
            </div>
          </div>
        </main>
        
        {/* ErrorFeedback display */}
        <FeedbackDisplay
          feedbackItems={feedbackItems}
          onDismiss={handleDismissFeedback}
          onRecoverySelect={handleRecoverySelect}
          isLightMode={isLightMode}
        />
      </div>
    </div>
  );
}

export default App;
