/**
 * AuditLogger Module (M16/M19)
 * Logs events and diagnostics for audit purposes
 */

class AuditLogger {
  constructor() {
    this.logs = [];
    this.maxLogs = 1000; // Keep last 1000 logs in memory
  }

  /**
   * Log an error event
   * @param {Object} event - ErrorEvent object
   */
  logError(event) {
    const logEntry = {
      type: 'error',
      ...event,
      loggedAt: new Date().toISOString()
    };
    
    this.addLog(logEntry);
    console.error('[AuditLogger] Error:', logEntry);
  }

  /**
   * Log a general event
   * @param {Object} event - Event object
   */
  logEvent(event) {
    const logEntry = {
      ...event,
      loggedAt: new Date().toISOString()
    };
    
    this.addLog(logEntry);
    console.log('[AuditLogger] Event:', logEntry);
  }

  /**
   * Add log entry (with rotation)
   * @param {Object} entry - Log entry
   */
  addLog(entry) {
    this.logs.push(entry);
    
    // Rotate if exceeds max
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    // In production, you might want to send to backend
    // this.sendToBackend(entry);
  }

  /**
   * Get recent logs
   * @param {number} count - Number of logs to retrieve
   * @returns {Array} Array of log entries
   */
  getRecentLogs(count = 100) {
    return this.logs.slice(-count);
  }

  /**
   * Get logs by type
   * @param {string} type - Log type filter
   * @returns {Array} Filtered log entries
   */
  getLogsByType(type) {
    return this.logs.filter(log => log.type === type);
  }

  /**
   * Clear all logs
   */
  clear() {
    this.logs = [];
  }
}

export default AuditLogger;
