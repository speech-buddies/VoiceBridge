// Background service worker for VoiceBridge extension

chrome.runtime.onInstalled.addListener(() => {
  console.log('VoiceBridge extension installed');
});

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'toggleExtension') {
    // Toggle extension visibility on the current tab
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'toggleWidget' });
      }
    });
  }
  return true;
});
