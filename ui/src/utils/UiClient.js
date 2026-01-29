/**
 * UiClient - Bridge between ErrorFeedback and React UI
 * Implements the notifier interface for ErrorFeedback
 */

class UiClient {
  constructor(setFeedbackItems, setError) {
    this.setFeedbackItems = setFeedbackItems;
    this.setError = setError;
    this.currentItems = [];
  }

  /**
   * Display an error feedback item
   * @param {Object} feedbackItem - FeedbackItem object
   */
  displayError(feedbackItem) {
    this.addFeedbackItem(feedbackItem);
  }

  /**
   * Display a recovery prompt
   * @param {Object} feedbackItem - FeedbackItem with recovery options
   */
  displayRecovery(feedbackItem) {
    this.addFeedbackItem(feedbackItem);
  }

  /**
   * Hide a feedback item
   * @param {string} feedbackId - UUID of feedback item
   */
  hideFeedback(feedbackId) {
    this.currentItems = this.currentItems.filter(item => item.id !== feedbackId);
    this.updateUI();
  }

  /**
   * Add feedback item to current items
   * @param {Object} feedbackItem - FeedbackItem object
   */
  addFeedbackItem(feedbackItem) {
    // Remove existing item with same ID if present
    this.currentItems = this.currentItems.filter(item => item.id !== feedbackItem.id);
    // Add new item
    this.currentItems.push(feedbackItem);
    this.updateUI();
  }

  /**
   * Update UI with current feedback items
   */
  updateUI() {
    if (this.setFeedbackItems) {
      this.setFeedbackItems([...this.currentItems]);
    }
  }

  /**
   * Clear all feedback items
   */
  clear() {
    this.currentItems = [];
    this.updateUI();
  }
}

export default UiClient;
