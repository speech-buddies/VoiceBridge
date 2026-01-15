import React, { useState } from 'react';
import './App.css';
import SpeechInput from './components/SpeechInput';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import CommandConfirmation from './components/CommandConfirmation';
import StatusIndicator from './components/StatusIndicator';

function App() {
  const [transcription, setTranscription] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [status, setStatus] = useState('idle'); // idle, listening, processing, confirmed, executed
  const [command, setCommand] = useState(null);
  const [error, setError] = useState(null);
  const [isLightMode, setIsLightMode] = useState(false);

  const handleStartListening = () => {
    setIsListening(true);
    setStatus('listening');
    setError(null);
    setTranscription('');
    setCommand(null);
  };

  const handleTranscription = (text) => {
    setTranscription(text);
    setStatus('processing');
  };

  const handleStopListening = () => {
    setIsListening(false);
    setStatus('processing');
    // Simulate processing delay
    setTimeout(() => {
      setStatus('idle');
    }, 1000);
  };

  const handleCommandDetected = (cmd) => {
    setCommand(cmd);
    setStatus('confirmed');
  };

  const handleConfirmCommand = () => {
    setStatus('executed');
    // Simulate command execution
    setTimeout(() => {
      setStatus('idle');
      setCommand(null);
      setTranscription('');
    }, 2000);
  };

  const handleCancelCommand = () => {
    setStatus('idle');
    setCommand(null);
    setTranscription('');
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setStatus('idle');
    setIsListening(false);
  };

  const toggleTheme = () => {
    setIsLightMode(!isLightMode);
  };

  return (
    <div className={`App ${isLightMode ? 'light-mode' : 'dark-mode'}`}>
      <header className="App-header">
        <div className="header-content">
          <div>
            <h1>VoiceBridge</h1>
            <p className="subtitle">Accessibility Interface for Speech-to-Text</p>
          </div>
          <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
            {isLightMode ? '🌙' : '☀️'}
          </button>
        </div>
      </header>

      <main className="App-main">
        <div className="main-container">
          <div className="left-panel">
            <StatusIndicator status={status} error={error} isLightMode={isLightMode} />
            <SpeechInput
              isListening={isListening}
              onStartListening={handleStartListening}
              onStopListening={handleStopListening}
              onTranscription={handleTranscription}
              onError={handleError}
              isLightMode={isLightMode}
            />
            {command && (
              <CommandConfirmation
                command={command}
                onConfirm={handleConfirmCommand}
                onCancel={handleCancelCommand}
                isLightMode={isLightMode}
              />
            )}
          </div>

          <div className="right-panel">
            <TranscriptionDisplay
              transcription={transcription}
              isListening={isListening}
              onCommandDetected={handleCommandDetected}
              isLightMode={isLightMode}
            />
          </div>
        </div>
      </main>

      <footer className="App-footer">
        <p>Speech Buddies - Making technology accessible for everyone</p>
      </footer>
    </div>
  );
}

export default App;
