// Popup script for VoiceBridge extension

document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('toggleBtn');
  const status = document.getElementById('status');

  // Check if we can inject content script on the current tab
  const checkAndInject = async (tabId) => {
    try {
      // Try to send a ping message first
      chrome.tabs.sendMessage(tabId, { action: 'ping' }, (response) => {
        if (chrome.runtime.lastError) {
          // Content script not loaded, try to inject it
          chrome.scripting.executeScript({
            target: { tabId: tabId },
            files: ['content.js']
          }, () => {
            if (chrome.runtime.lastError) {
              status.textContent = 'Cannot inject on this page. Try a regular website.';
              status.style.color = '#ff6b6b';
            } else {
              // Wait a bit for script to load, then toggle
              setTimeout(() => {
                toggleWidget(tabId);
              }, 200);
            }
          });
        } else {
          // Content script is ready, toggle widget
          toggleWidget(tabId);
        }
      });
    } catch (error) {
      status.textContent = 'Error: ' + error.message;
      status.style.color = '#ff6b6b';
    }
  };

  const toggleWidget = (tabId) => {
    chrome.tabs.sendMessage(tabId, { action: 'toggleWidget' }, (response) => {
      if (chrome.runtime.lastError) {
        status.textContent = 'Error: ' + chrome.runtime.lastError.message;
        status.style.color = '#ff6b6b';
      } else {
        status.textContent = 'Widget toggled on the current page.';
        status.style.color = '#a0a0a0';
      }
    });
  };

  toggleBtn.addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        // Check if it's a special page where we can't inject
        const url = tabs[0].url;
        if (url.startsWith('chrome://') || url.startsWith('chrome-extension://') || url.startsWith('edge://')) {
          status.textContent = 'Cannot use on this page. Navigate to a regular website.';
          status.style.color = '#ff6b6b';
          return;
        }
        checkAndInject(tabs[0].id);
      }
    });
  });
});
