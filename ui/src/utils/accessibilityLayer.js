/**
 * Accessibility Layer Module (M2)
 * Ensures error messages and prompts are announced accessibly
 */

class AccessibilityLayer {
  constructor() {
    this.announcementContainer = null;
    this.initAnnouncementContainer();
  }

  /**
   * Initialize hidden announcement container for screen readers
   */
  initAnnouncementContainer() {
    if (typeof document !== 'undefined') {
      this.announcementContainer = document.createElement('div');
      this.announcementContainer.setAttribute('role', 'status');
      this.announcementContainer.setAttribute('aria-live', 'polite');
      this.announcementContainer.setAttribute('aria-atomic', 'true');
      this.announcementContainer.className = 'sr-only';
      this.announcementContainer.style.cssText = `
        position: absolute;
        left: -10000px;
        width: 1px;
        height: 1px;
        overflow: hidden;
      `;
      document.body.appendChild(this.announcementContainer);
    }
  }

  /**
   * Announce a message to screen readers
   * @param {string} message - Message to announce
   * @param {string} priority - 'polite' | 'assertive' | 'alert' | 'status'
   */
  announce(message, priority = 'polite') {
    if (!this.announcementContainer) {
      this.initAnnouncementContainer();
    }

    if (!this.announcementContainer) {
      // Fallback if DOM not available
      console.log('[A11y]', message);
      return;
    }

    // Set appropriate aria-live attribute
    const liveValue = priority === 'alert' || priority === 'assertive' ? 'assertive' : 'polite';
    this.announcementContainer.setAttribute('aria-live', liveValue);

    // Clear and set new message
    this.announcementContainer.textContent = '';
    setTimeout(() => {
      this.announcementContainer.textContent = message;
    }, 100);
  }

  /**
   * Announce error message
   * @param {string} message - Error message
   */
  announceError(message) {
    this.announce(message, 'alert');
  }

  /**
   * Announce status update
   * @param {string} message - Status message
   */
  announceStatus(message) {
    this.announce(message, 'status');
  }
}

export default AccessibilityLayer;
