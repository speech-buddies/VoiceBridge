from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pathlib import Path

app = FastAPI()

# Allow calls from the extension / browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder where audio chunks will be saved (inside this backend project)
SAVE_DIR = Path(__file__).resolve().parent / "Recordings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/audio")
async def receive_audio(
    audio: UploadFile = File(...),
    mimeType: str = Form("audio/webm"),
    timestamp: str = Form(...),
):
    """
    Receives a single audio chunk from the extension and writes it to disk.
    """
    suffix = Path(audio.filename).suffix or ".webm"
    filename = SAVE_DIR / f"chunk-{timestamp}{suffix}"

    with filename.open("wb") as f:
        f.write(await audio.read())

    return PlainTextResponse("ok")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

