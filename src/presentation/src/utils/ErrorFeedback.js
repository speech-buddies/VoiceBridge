/**
 * ErrorFeedback Module (M11)
 * Manages error notifications, recovery prompts, and error logging
 */

import { v4 as uuidv4 } from 'uuid';

// Error message templates (ERROR_COPY_PATH equivalent)
const ERROR_MESSAGES = {
  'MIC_PERMISSION_DENIED': {
    title: 'Microphone Access Denied',
    message: 'VoiceBridge needs microphone access to function. Please enable it in your browser settings.',
    severity: 'high',
    recoveryOptions: [
      'Open browser settings',
      'Check microphone permissions',
      'Reload extension'
    ]
  },
  'NETWORK_ERROR': {
    title: 'Network Connection Failed',
    message: 'Unable to send audio to server. Check your connection and try again.',
    severity: 'medium',
    recoveryOptions: [
      'Retry connection',
      'Check server status',
      'Continue offline'
    ]
  },
  'AUDIO_DEVICE_ERROR': {
    title: 'Audio Device Error',
    message: 'Microphone is not available or already in use by another application.',
    severity: 'high',
    recoveryOptions: [
      'Close other applications',
      'Check microphone connection',
      'Select different microphone'
    ]
  },
  'BACKEND_ERROR': {
    title: 'Backend Server Error',
    message: 'The audio processing server encountered an error.',
    severity: 'medium',
    recoveryOptions: [
      'Retry',
      'Check server logs',
      'Restart server'
    ]
  },
  'GENERIC_ERROR': {
    title: 'Error',
    message: 'An unexpected error occurred.',
    severity: 'low',
    recoveryOptions: [
      'Retry',
      'Report issue'
    ]
  }
};

/**
 * Format error message from code and detail
 * @param {string} code - Error code
 * @param {string} detail - Error detail message
 * @returns {string} Formatted user-facing error message
 */
const formatErrorMessage = (code, detail) => {
  const template = ERROR_MESSAGES[code] || ERROR_MESSAGES['GENERIC_ERROR'];
  return `${template.title}: ${detail || template.message}`;
};

/**
 * Create recovery options object from string list
 * @param {string[]} options - List of recovery option strings
 * @returns {Object} RecoveryOptions object
 */
const makeRecovery = (options) => {
  return {
    options: options.map((opt, index) => ({
      id: uuidv4(),
      label: opt,
      action: null // Will be set by caller
    }))
  };
};

/**
 * ErrorFeedback class - manages error notifications and recovery prompts
 */
class ErrorFeedback {
  constructor(notifier, auditLogger = null, accessibilityLayer = null) {
    this.active = new Map(); // Map<UUID, FeedbackItem>
    this.notifier = notifier; // UiClient handle
    this.auditLogger = auditLogger;
    this.accessibilityLayer = accessibilityLayer;
    this.defaultLang = navigator.language || 'en-US';
  }

  /**
   * Show an error notification
   * @param {string} code - Error code
   * @param {string} detail - Error detail message
   * @returns {string} feedbackId (UUID) or null if duplicate
   */
  showError(code, detail) {
    // Check if same error already exists (deduplication)
    const existingError = Array.from(this.active.values()).find(
      item => item.type === 'error' && item.code === code && item.detail === detail
    );
    
    if (existingError) {
      // Error already shown, return existing ID
      return existingError.id;
    }

    const feedbackId = uuidv4();
    const formattedMessage = formatErrorMessage(code, detail);
    const template = ERROR_MESSAGES[code] || ERROR_MESSAGES['GENERIC_ERROR'];

    const feedbackItem = {
      id: feedbackId,
      type: 'error',
      code,
      message: formattedMessage,
      detail,
      severity: template.severity,
      timestamp: new Date().toISOString(),
      recoveryOptions: template.recoveryOptions ? makeRecovery(template.recoveryOptions) : null
    };

    this.active.set(feedbackId, feedbackItem);

    // Display via notifier (UI client)
    if (this.notifier) {
      this.notifier.displayError(feedbackItem);
    }

    // Announce accessibly
    if (this.accessibilityLayer) {
      this.accessibilityLayer.announce(formattedMessage, 'alert');
    }

    // Log error event
    if (this.auditLogger) {
      this.log({
        type: 'error',
        code,
        detail,
        feedbackId,
        timestamp: feedbackItem.timestamp,
        severity: template.severity
      });
    }

    return feedbackId;
  }

  /**
   * Show recovery prompt with options
   * @param {string} cmdId - Command ID (UUID)
   * @param {string[]} options - List of recovery option strings
   * @returns {string} feedbackId (UUID)
   */
  showRecovery(cmdId, options) {
    const feedbackId = uuidv4();
    const recoveryOptions = makeRecovery(options);

    const feedbackItem = {
      id: feedbackId,
      type: 'recovery',
      cmdId,
      recoveryOptions,
      timestamp: new Date().toISOString()
    };

    this.active.set(feedbackId, feedbackItem);

    // Display recovery prompt via notifier
    if (this.notifier) {
      this.notifier.displayRecovery(feedbackItem);
    }

    // Announce accessibly
    if (this.accessibilityLayer) {
      this.accessibilityLayer.announce(
        `Recovery options available: ${options.join(', ')}`,
        'status'
      );
    }

    return feedbackId;
  }

  /**
   * Dismiss a feedback item
   * @param {string} feedbackId - UUID of feedback item
   * @returns {boolean} True if removed, false otherwise
   */
  dismiss(feedbackId) {
    const feedbackItem = this.active.get(feedbackId);
    if (!feedbackItem) {
      return false;
    }

    this.active.delete(feedbackId);

    // Hide in UI
    if (this.notifier) {
      this.notifier.hideFeedback(feedbackId);
    }

    return true;
  }

  /**
   * Log an error event to audit log
   * @param {Object} event - ErrorEvent object
   */
  log(event) {
    if (this.auditLogger) {
      this.auditLogger.logError({
        ...event,
        timestamp: event.timestamp || new Date().toISOString()
      });
    } else {
      // Fallback to console if no logger
      console.error('ErrorFeedback log:', event);
    }
  }

  /**
   * Get all active feedback items
   * @returns {Array} Array of active FeedbackItem objects
   */
  getActive() {
    return Array.from(this.active.values());
  }

  /**
   * Clear all active feedback items
   */
  clearAll() {
    const ids = Array.from(this.active.keys());
    ids.forEach(id => this.dismiss(id));
  }
}

export default ErrorFeedback;
export { ERROR_MESSAGES, formatErrorMessage, makeRecovery };
