from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pathlib import Path
import sys
import wave
import threading

# Add project root so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

# Lazy import: audio_capture needs sounddevice + webrtcvad
def _get_audio_capture():
    try:
        from input.audio_capture import RealtimeAudioCapture, AudioConfig
        return RealtimeAudioCapture, AudioConfig, None
    except ImportError as e:
        return None, None, str(e)

app = FastAPI()

# Allow calls from the extension / browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder where audio recordings are saved
SAVE_DIR = Path(__file__).resolve().parent / "Recordings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

_capture_instance = None
_capture_lock = threading.Lock()


def _on_audio_ready(audio_float32: np.ndarray, metadata: dict):
    """Save captured audio (from RealtimeAudioCapture) as WAV to Recordings/."""
    import time
    timestamp = int(time.time() * 1000)
    filename = SAVE_DIR / f"chunk-{timestamp}.wav"
    audio_int16 = (audio_float32 * 32767).astype(np.int16)
    sample_rate = metadata.get("sample_rate", 16000)
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


@app.post("/audio/capture/start")
async def start_audio_capture():
    """Start backend-driven audio capture using audio_capture.py."""
    global _capture_instance
    RealtimeAudioCapture, AudioConfig, import_err = _get_audio_capture()
    if import_err:
        err = str(import_err)
        if "pkg_resources" in err:
            hint = "Run: pip install setuptools"
        elif "webrtcvad" in err:
            hint = "Run: pip install webrtcvad"
        elif "sounddevice" in err:
            hint = "Run: pip install sounddevice"
        else:
            hint = "Run: pip install -r requirements.txt"
        return {"ok": False, "message": f"Audio capture unavailable: {import_err}. {hint}"}
    with _capture_lock:
        if _capture_instance is not None:
            return {"ok": True, "message": "Already capturing"}
        try:
            config = AudioConfig(sample_rate=16000, vad_aggressiveness=3)
            _capture_instance = RealtimeAudioCapture(config=config, on_audio_ready=_on_audio_ready)
            _capture_instance.start()
            return {"ok": True, "message": "Audio capture started"}
        except Exception as e:
            return {"ok": False, "message": str(e)}


@app.post("/audio/capture/stop")
async def stop_audio_capture():
    """Stop backend-driven audio capture."""
    global _capture_instance
    with _capture_lock:
        if _capture_instance is None:
            return {"ok": True, "message": "Not capturing"}
        try:
            _capture_instance.stop()
            _capture_instance = None
            return {"ok": True, "message": "Audio capture stopped"}
        except Exception as e:
            _capture_instance = None
            return {"ok": False, "message": str(e)}


@app.get("/audio/capture/status")
async def audio_capture_status():
    """Get status of backend audio capture."""
    with _capture_lock:
        if _capture_instance is None:
            return {"capturing": False, "state": "idle"}
        return {"capturing": True, "state": _capture_instance.state, "stats": _capture_instance.get_stats()}


@app.post("/audio")
async def receive_audio(
    audio: UploadFile = File(...),
    mimeType: str = Form("audio/webm"),
    timestamp: str = Form(...),
):
    """Legacy: receives audio blob from browser (when not using backend capture)."""
    suffix = Path(audio.filename).suffix or ".webm"
    filename = SAVE_DIR / f"chunk-{timestamp}{suffix}"
    with filename.open("wb") as f:
        f.write(await audio.read())
    return PlainTextResponse("ok")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

