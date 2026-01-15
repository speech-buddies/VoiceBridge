import React from 'react';
import './CommandConfirmation.css';

const CommandConfirmation = ({ command, onConfirm, onCancel, isLightMode }) => {
  const getCommandDescription = () => {
    const actionMap = {
      new_tab: 'Open a new browser tab',
      close_tab: 'Close the current tab',
      back: 'Go back in browser history',
      forward: 'Go forward in browser history',
      refresh: 'Refresh the current page',
      scroll_down: 'Scroll down the page',
      scroll_up: 'Scroll up the page',
    };
    return actionMap[command.action] || `Execute: ${command.action}`;
  };

  return (
    <div className={`command-confirmation ${isLightMode ? 'light-mode' : 'dark-mode'}`}>
      <h2>Command Detected</h2>
      <div className="command-box">
        <div className="command-icon">⚡</div>
        <p className="command-description">{getCommandDescription()}</p>
        <p className="command-original">"{command.originalText}"</p>
      </div>
      <div className="confirmation-buttons">
        <button
          className="confirm-button"
          onClick={onConfirm}
          aria-label="Confirm and execute command"
        >
          ✓ Confirm & Execute
        </button>
        <button
          className="cancel-button"
          onClick={onCancel}
          aria-label="Cancel command"
        >
          ✗ Cancel
        </button>
      </div>
    </div>
  );
};

export default CommandConfirmation;
