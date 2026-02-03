import React, { useState, useRef, useEffect, useCallback } from 'react';
import './SpeechInput.css';
import { backendFetch } from '../utils/backendApi';

const SpeechInput = ({
  isListening,
  onAudioData,
  onError,
  isLightMode,
  backendBase = 'http://localhost:8000',
}) => {
  const [audioLevel, setAudioLevel] = useState(0);
  const [captureState, setCaptureState] = useState('idle');
  const statusPollRef = useRef(null);

  const startBackendCapture = useCallback(async () => {
    try {
      const res = await backendFetch(`${backendBase}/audio/capture/start`, { method: 'POST' });
      const data = await res.json();
      if (!data.ok) {
        throw new Error(data.message || 'Failed to start audio capture');
      }
    } catch (err) {
      console.error('Error starting backend audio capture:', err);
      onError(err.message || 'Unable to start audio capture. Is the backend running?');
    }
  }, [backendBase, onError]);

  const stopBackendCapture = useCallback(async () => {
    try {
      await backendFetch(`${backendBase}/audio/capture/stop`, { method: 'POST' });
    } catch (err) {
      console.error('Error stopping backend audio capture:', err);
    }
  }, [backendBase]);

  const pollStatus = useCallback(() => {
    const poll = async () => {
      try {
        const res = await backendFetch(`${backendBase}/audio/capture/status`);
        const data = await res.json();
        setCaptureState(data.state || (data.capturing ? 'listening' : 'idle'));
        if (data.state === 'recording' || data.state === 'speech_detected') {
          setAudioLevel(60);
        } else if (data.capturing) {
          setAudioLevel(20);
        } else {
          setAudioLevel(0);
        }
      } catch {
        setCaptureState('idle');
        setAudioLevel(0);
      }
    };
    poll();
    statusPollRef.current = setInterval(poll, 500);
  }, [backendBase]);

  useEffect(() => {
    if (isListening) {
      startBackendCapture();
      pollStatus();
    } else {
      stopBackendCapture();
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current);
        statusPollRef.current = null;
      }
      setCaptureState('idle');
      setAudioLevel(0);
    }

    return () => {
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current);
      }
    };
  }, [isListening, startBackendCapture, stopBackendCapture, pollStatus]);

  return (
    <div className={`speech-input ${isLightMode ? 'light-mode' : 'dark-mode'}`}>
      <h2>Speech Input</h2>
      <p className="backend-capture-hint">Using backend audio capture (saves to backend/Recordings)</p>
      <div className="microphone-container">
        <button
          className={`microphone-button ${isListening ? 'listening' : ''}`}
          aria-label="Listening for speech"
          aria-pressed={isListening}
        >
          <svg
            width="80"
            height="80"
            viewBox="0 0 80 80"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M40 20C35.58 20 32 23.58 32 28V40C32 44.42 35.58 48 40 48C44.42 48 48 44.42 48 40V28C48 23.58 44.42 20 40 20Z"
              fill={isLightMode ? '#000000' : '#ffffff'}
            />
            <path
              d="M50 40C50 45.52 45.52 50 40 50C34.48 50 30 45.52 30 40H26C26 47.73 32.27 54 40 54C47.73 54 54 47.73 54 40H50Z"
              fill={isLightMode ? '#000000' : '#ffffff'}
            />
            <path
              d="M40 56V60"
              stroke={isLightMode ? '#000000' : '#ffffff'}
              strokeWidth="3"
              strokeLinecap="round"
            />
          </svg>
          {isListening && (
            <div className="audio-wave" style={{ '--audio-level': audioLevel }} />
          )}
        </button>
        <p className="status-text">
          {!isListening
            ? 'Click to start (recordings save to backend/Recordings)'
            : captureState === 'recording' || captureState === 'speech_detected'
            ? 'Recording...'
            : 'Listening... Speak your command'}
        </p>
      </div>
    </div>
  );
};

export default SpeechInput;
