// Content script to inject VoiceBridge widget into web pages

(function() {
  'use strict';

  // Check if widget is already injected
  if (document.getElementById('voicebridge-root')) {
    return;
  }

  // Create container for React app
  const root = document.createElement('div');
  root.id = 'voicebridge-root';
  root.setAttribute('data-voicebridge', 'true');
  root.style.display = 'none'; // Start hidden
  document.body.appendChild(root);

  // Inject React app styles and scripts
  const injectScript = (src) => {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = chrome.runtime.getURL(src);
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  };

  const injectStyle = (href) => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = chrome.runtime.getURL(href);
    document.head.appendChild(link);
  };

  // Load React app
  const loadApp = async () => {
    try {
      // Try to load asset manifest first
      let assetManifest = null;
      try {
        const manifestResponse = await fetch(chrome.runtime.getURL('asset-manifest.json'));
        if (manifestResponse.ok) {
          assetManifest = await manifestResponse.json();
        }
      } catch (e) {
        console.log('No asset manifest found, using default paths');
      }
      
      // Inject CSS files
      if (assetManifest && assetManifest.css && assetManifest.css.length > 0) {
        assetManifest.css.forEach(injectStyle);
      } else {
        // Fallback: try common CSS paths
        const cssFiles = [
          'static/css/main.css',
          'static/css/main.*.css'
        ];
        cssFiles.forEach(file => {
          if (file.includes('*')) {
            // For wildcard patterns, we'd need to list files, but for now just try the common one
            injectStyle('static/css/main.css');
          } else {
            injectStyle(file);
          }
        });
      }
      
      // Inject JavaScript bundles
      if (assetManifest && assetManifest.js && assetManifest.js.length > 0) {
        // Load all JS files in order
        for (const script of assetManifest.js) {
          try {
            await injectScript(script);
          } catch (e) {
            console.warn(`Failed to load ${script}:`, e);
          }
        }
      } else {
        // Fallback: try common JS paths
        const scriptFiles = [
          'static/js/main.js',
          'static/js/bundle.js'
        ];
        
        for (const script of scriptFiles) {
          try {
            await injectScript(script);
            break;
          } catch (e) {
            continue;
          }
        }
      }
    } catch (error) {
      console.error('Error loading VoiceBridge:', error);
    }
  };

  // Listen for messages from popup/background
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'ping') {
      // Respond to ping to confirm content script is loaded
      sendResponse({ loaded: true });
      return true;
    }
    
    if (request.action === 'toggleWidget') {
      const widget = document.getElementById('voicebridge-root');
      if (widget) {
        const isHidden = widget.style.display === 'none' || !widget.style.display;
        widget.style.display = isHidden ? 'block' : 'none';
        sendResponse({ visible: !isHidden });
      } else {
        // Widget not created yet, try to load it
        loadApp().then(() => {
          const widget = document.getElementById('voicebridge-root');
          if (widget) {
            widget.style.display = 'block';
            sendResponse({ visible: true });
          } else {
            sendResponse({ visible: false, error: 'Widget not found' });
          }
        });
        return true; // Keep channel open for async response
      }
      return true; // Keep channel open for async response
    }
    return false;
  });

  // Load the app when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadApp);
  } else {
    loadApp();
  }
})();
