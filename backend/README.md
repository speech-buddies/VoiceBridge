## VoiceBridge Backend

This backend receives 10-second audio chunks from the VoiceBridge browser extension and saves them as `.webm` files in the local `Recordings` folder.

### Folder structure

- `server.py`: FastAPI app that exposes a `POST /audio` endpoint.
- `requirements.txt`: Python dependencies for the backend.
- `Recordings/`: Folder where incoming audio chunks are stored (created automatically on first run).

### Prerequisites

- Python 3.9+ installed

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

### Running the server

From the `backend` folder with the virtual environment activated:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

You should see a message similar to:

```text
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Testing the endpoint

Open your browser and go to:

   ```text
   http://localhost:8000/docs
   ```

In the UI code (`ui/src/App.js`), the extension is configured to send each ~10 second audio chunk to:

```js
const AUDIO_BACKEND_URL = 'http://localhost:8000/audio';
```

With the server running and the extension active, new `.webm` audio files will appear continuously in `backend/Recordings` while you are speaking.

