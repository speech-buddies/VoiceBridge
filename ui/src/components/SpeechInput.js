import React, { useState, useRef, useEffect } from 'react';
import './SpeechInput.css';

const SpeechInput = ({
  isListening,
  onStartListening,
  onStopListening,
  onTranscription,
  onError,
  isLightMode,
}) => {
  const [audioLevel, setAudioLevel] = useState(0);
  const mediaStreamRef = useRef(null);
  const animationFrameRef = useRef(null);

  useEffect(() => {
    if (isListening) {
      startMicrophone();
    } else {
      stopMicrophone();
    }

    return () => {
      stopMicrophone();
    };
  }, [isListening]);

  const startMicrophone = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioContext.createAnalyser();
      const microphone = audioContext.createMediaStreamSource(stream);
      microphone.connect(analyser);

      analyser.fftSize = 256;
      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const updateAudioLevel = () => {
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        setAudioLevel(average);
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
      };

      updateAudioLevel();
    } catch (error) {
      console.error('Error accessing microphone:', error);
      onError('Unable to access microphone. Please check permissions.');
    }
  };

  const stopMicrophone = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setAudioLevel(0);
  };

  const handleToggleListening = () => {
    if (isListening) {
      onStopListening();
      // Simulate transcription after speaking stops
      setTimeout(() => {
        const sampleTexts = [
          'Open new tab',
          'Close tab',
          'Go back',
          'Go forward',
          'Refresh page',
          'Scroll down',
          'Scroll up',
        ];
        const randomText = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
        onTranscription(randomText);
      }, 500);
    } else {
      onStartListening();
    }
  };

  return (
    <div className={`speech-input ${isLightMode ? 'light-mode' : 'dark-mode'}`}>
      <h2>Speech Input</h2>
      <div className="microphone-container">
        <button
          className={`microphone-button ${isListening ? 'listening' : ''}`}
          onClick={handleToggleListening}
          aria-label={isListening ? 'Stop listening' : 'Start listening'}
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
            <div
              className="audio-wave"
              style={{ '--audio-level': audioLevel }}
            />
          )}
        </button>
        <p className="status-text">
          {isListening
            ? 'Listening... Speak your command'
            : 'Click to start speaking'}
        </p>
      </div>
    </div>
  );
};

export default SpeechInput;
