/**
 * FeedbackDisplay Component (M3)
 * Displays error notifications and recovery prompts
 */

import React, { useState, useEffect } from 'react';
import './FeedbackDisplay.css';
import { ERROR_MESSAGES } from '../utils/ErrorFeedback';

const FeedbackDisplay = ({ feedbackItems, onDismiss, onRecoverySelect, isLightMode }) => {
  const [visibleItems, setVisibleItems] = useState([]);

  useEffect(() => {
    setVisibleItems(feedbackItems);
  }, [feedbackItems]);

  const handleDismiss = (feedbackId) => {
    if (onDismiss) {
      onDismiss(feedbackId);
    }
  };

  const handleRecoverySelect = (feedbackId, optionId) => {
    if (onRecoverySelect) {
      onRecoverySelect(feedbackId, optionId);
    }
  };

  if (visibleItems.length === 0) {
    return null;
  }

  return (
    <div className={`feedback-container ${isLightMode ? 'light-mode' : 'dark-mode'}`}>
      {visibleItems.map((item) => (
        <div
          key={item.id}
          className={`feedback-item feedback-${item.type} feedback-severity-${item.severity || 'low'}`}
          role={item.type === 'error' ? 'alert' : 'dialog'}
          aria-live={item.type === 'error' ? 'assertive' : 'polite'}
        >
          <div className="feedback-header">
            <span className="feedback-icon">
              {item.type === 'error' ? '⚠️' : '💡'}
            </span>
            <span className="feedback-title">
              {item.type === 'error' 
                ? (ERROR_MESSAGES[item.code]?.title || 'Error')
                : 'Recovery Options'}
            </span>
            <button
              className="feedback-dismiss"
              onClick={() => handleDismiss(item.id)}
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>

          <div className="feedback-message">
            {item.message || item.detail}
          </div>

          {item.recoveryOptions && item.recoveryOptions.options && (
            <div className="feedback-recovery">
              <p className="recovery-label">Recovery options:</p>
              <div className="recovery-options">
                {item.recoveryOptions.options.map((option) => (
                  <button
                    key={option.id}
                    className="recovery-option"
                    onClick={() => handleRecoverySelect(item.id, option.id)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default FeedbackDisplay;
