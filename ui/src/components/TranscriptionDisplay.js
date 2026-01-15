import React, { useEffect, useRef } from 'react';
import './TranscriptionDisplay.css';

const TranscriptionDisplay = ({ transcription, isListening, onCommandDetected, isLightMode }) => {
  const textEndRef = useRef(null);

  useEffect(() => {
    // Simple command detection logic
    if (transcription) {
      const lowerText = transcription.toLowerCase();
      const commands = {
        'open new tab': { type: 'browser', action: 'new_tab' },
        'close tab': { type: 'browser', action: 'close_tab' },
        'go back': { type: 'browser', action: 'back' },
        'go forward': { type: 'browser', action: 'forward' },
        'refresh': { type: 'browser', action: 'refresh' },
        'scroll down': { type: 'browser', action: 'scroll_down' },
        'scroll up': { type: 'browser', action: 'scroll_up' },
      };

      for (const [key, command] of Object.entries(commands)) {
        if (lowerText.includes(key)) {
          onCommandDetected({ ...command, originalText: transcription });
          return;
        }
      }
    }
  }, [transcription, onCommandDetected]);

  useEffect(() => {
    // Auto-scroll to bottom when new text appears
    if (textEndRef.current) {
      textEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [transcription]);

  return (
    <div className={`transcription-display ${isLightMode ? 'light-mode' : 'dark-mode'}`}>
      <h2>Transcription</h2>
      <div className="transcription-box">
        <div className="transcription-content">
          <p className="transcription-text">
            {transcription || (
              <span className="placeholder-text">
                {isListening ? 'Listening...' : 'Your transcription will appear here after you finish speaking'}
              </span>
            )}
          </p>
          <div ref={textEndRef} />
        </div>
      </div>
      {transcription && (
        <p className="transcription-hint">
          Review the transcription. If a command is detected, you'll be asked to confirm.
        </p>
      )}
    </div>
  );
};

export default TranscriptionDisplay;
