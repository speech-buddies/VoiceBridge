import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import SpeechInput from './components/SpeechInput';
import StatusIndicator from './components/StatusIndicator';
import FeedbackDisplay from './components/FeedbackDisplay';
import ErrorFeedback from './utils/ErrorFeedback';
import UiClient from './utils/UiClient';
import AuditLogger from './utils/auditLogger';
import AccessibilityLayer from './utils/accessibilityLayer';

const AUDIO_BACKEND_BASE = 'http://localhost:8000';

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

  // Tell backend to start/stop audio capture (server records and saves to Recordings/)
  useEffect(() => {
    const base = AUDIO_BACKEND_BASE;
    if (isListening) {
      fetch(`${base}/audio/capture/start`, { method: 'POST' })
        .then((r) => r.json())
        .then((data) => {
          if (!data.ok) console.warn('Backend capture start:', data.message);
        })
        .catch((err) => console.warn('Failed to start backend capture:', err));
    } else {
      fetch(`${base}/audio/capture/stop`, { method: 'POST' }).catch(() => {});
    }
    return () => {
      fetch(`${base}/audio/capture/stop`, { method: 'POST' }).catch(() => {});
    };
  }, [isListening]);

  // Poll backend capture status to show Recording when user is speaking
  useEffect(() => {
    if (!isListening) return;
    const base = AUDIO_BACKEND_BASE;
    const interval = setInterval(() => {
      fetch(`${base}/audio/capture/status`)
        .then((r) => r.json())
        .then((data) => {
          if (data.capturing && data.state) {
            if (data.state === 'speech_detected' || data.state === 'recording') {
              setStatus('recording');
            } else if (data.state === 'processing') {
              setStatus('processing');
            } else {
              setStatus('listening');
            }
          }
        })
        .catch(() => {});
    }, 400);
    return () => clearInterval(interval);
  }, [isListening]);

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

  const handleAudioData = async () => {
    // Server (src/server.py) captures and saves to Recordings/; status kept by polling /audio/capture/status
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
              <p className="subtitle">An accessibility interface</p>
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
                status={status}
                onAudioData={handleAudioData}
                onError={handleError}
                isLightMode={isLightMode}
                backendBase={AUDIO_BACKEND_BASE}
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
