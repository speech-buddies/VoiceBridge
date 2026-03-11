"""
TrainingDataRecorder — saves confirmed voice command samples for training.
Audio: data/training/audio/*.wav, Manifest: data/training/samples.jsonl
Samples are only added when custom_training_enabled is True.
"""

import json
import threading
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger("TrainingDataRecorder")

_DATA_DIR    = Path(__file__).resolve().parent.parent.parent / "data" / "training"
_AUDIO_DIR   = _DATA_DIR / "audio"
_MANIFEST    = _DATA_DIR / "samples.jsonl"

_lock = threading.RLock()


def save_sample(
    audio_bytes: bytes,
    transcript: str,
    sample_rate: int = 16000,
    *,
    audio_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> float:
    """
    Save a confirmed command sample and return its duration in seconds.

    Parameters
    ----------
    audio_bytes   : raw int16 PCM bytes from on_silence_detected
    transcript    : the confirmed command text
    sample_rate   : sample rate of the audio (default 16000)
    audio_dir     : override for testing
    manifest_path : override for testing

    Returns
    -------
    duration_s : float — audio duration, for updating accumulated_audio_seconds
    """
    a_dir = Path(audio_dir) if audio_dir else _AUDIO_DIR
    m_path = Path(manifest_path) if manifest_path else _MANIFEST

    sample_id  = uuid.uuid4().hex
    wav_name   = f"{sample_id}.wav"
    wav_path   = a_dir / wav_name
    num_frames = len(audio_bytes) // 2          # int16 = 2 bytes per sample
    duration_s = num_frames / sample_rate

    with _lock:
        a_dir.mkdir(parents=True, exist_ok=True)
        m_path.parent.mkdir(parents=True, exist_ok=True)

        # Write WAV
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)              # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)

        # Append manifest entry
        entry = {
            "id":         sample_id,
            "transcript": transcript,
            "wav_file":   wav_name,
            "duration_s": round(duration_s, 3),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
        with m_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        "Training sample saved: %s | %.2fs | %s",
        wav_name, duration_s, transcript[:60],
    )
    return duration_s