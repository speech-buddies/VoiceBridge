# VoiceBridge UI

Browser extension UI for VoiceBridge - an accessibility interface for speech-to-text and browser control.

## Overview

This is a React-based browser extension that provides a user interface for continuous audio capture and speech recognition.

## Features

- **Speech Input**: Continuous audio capture and streaming to backend
- **Status Indicators**: Visual feedback for listening state and errors
- **Error Handling**: User-friendly error messages with recovery options
- **Accessibility**: Keyboard shortcuts and screen reader support
- **Theme Support**: Light and dark mode toggle

## Structure

```
presentation/
├── src/                    # React application source code
│   ├── components/         # UI components (SpeechInput, StatusIndicator, FeedbackDisplay)
│   ├── utils/              # Utility modules (UiClient, ErrorFeedback, AuditLogger)
│   └── App.js              # Main application component
├── background.js           # Extension service worker
├── content.js              # Content script for page interaction
├── popup.html              # Extension popup interface
├── manifest.json           # Extension manifest (Manifest V3)
└── scripts/                # Build scripts
```

## Setup

1. Install dependencies:

```bash
npm install
```

2. Build the extension:

```bash
npm run build:extension
```

3. Load the extension in Chrome:
   - Open `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `ui` folder

## Development

- **Start development server**: `npm start`
- **Build for production**: `npm run build`
- **Build extension**: `npm run build:extension`

## Configuration

The backend URL is configured in `src/App.js`. By default, it points to `http://localhost:8000/audio`.

## Keyboard Shortcuts

- `Alt+L`: Toggle light/dark mode
- `Escape`: Collapse the widget
