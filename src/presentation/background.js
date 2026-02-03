// Background service worker for VoiceBridge extension

chrome.runtime.onInstalled.addListener(() => {
  console.log('VoiceBridge extension installed');
});

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'toggleExtension') {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'toggleWidget' });
      }
    });
    return true;
  }
  // Proxy backend fetches to bypass page CSP (e.g. ChatGPT blocks localhost)
  if (request.action === 'backendFetch') {
    const { url, method = 'GET' } = request;
    fetch(url, { method })
      .then((res) => res.json().catch(() => ({})))
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }
  return true;
});
