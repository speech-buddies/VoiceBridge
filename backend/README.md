## VoiceBridge Backend

This backend records audio using `src/input/audio_capture.py` (RealtimeAudioCapture with WebRTC VAD). The UI triggers start/stop via REST; the backend captures from the system microphone and saves WAV files to the local `Recordings` folder.

### Folder structure

- `server.py` – FastAPI app with capture endpoints
- `requirements.txt` – Python dependencies
- `Recordings/` – Saved audio recordings (created automatically on first run)

### Endpoints

- `POST /audio/capture/start` – Start backend-driven audio capture (uses audio_capture.py)
- `POST /audio/capture/stop` – Stop audio capture
- `GET /audio/capture/status` – Get capture state (idle, listening, recording, etc.)
- `POST /audio` – Legacy endpoint for browser-recorded WebM blobs

### Prerequisites

- Python 3.9+
- Microphone access on the host machine

### Setup

1. Open a terminal in the `backend` folder:

   ```bash
   cd backend
   ```

2. Create and activate a virtual environment (Windows):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Includes: `numpy`, `sounddevice`, `webrtcvad`, `setuptools`.

### Running the server

From the `backend` folder with the virtual environment activated:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

You should see:

```text
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### How it works

1. The UI calls `POST /audio/capture/start` when the user enables listening
2. The backend starts `RealtimeAudioCapture` and records from the system microphone
3. Voice activity detection (VAD) detects speech; recordings end after ~1.5s of silence or 30s max
4. Each utterance is saved as `chunk-{timestamp}.wav` (16 kHz mono) in `Recordings/`
5. The UI calls `POST /audio/capture/stop` when the user disables listening

### Testing

- **API docs:** http://localhost:8000/docs
- **Capture status:** http://localhost:8000/audio/capture/status

With the server running and the extension active, new `.wav` files will appear in `backend/Recordings/` as you speak.
