import soundfile as sf
import numpy as np
from src.models.audio_data import AudioStream

def wav_to_audiostream(wav_path: str) -> AudioStream:
    """
    Load a WAV file and convert it to an AudioStream instance.
    
    Args:
        wav_path (str): Path to the WAV file.
    
    Returns:
        AudioStream: Dataclass with `samples` (float32) and `sample_rate`.
    """
    # Read WAV file
    samples, sample_rate = sf.read(wav_path)

    # Ensure mono
    if samples.ndim > 1:
        samples = samples[:, 0]

    # Convert to float32
    if samples.dtype != np.float32:
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            samples = samples.astype(np.float32) / 2**31
        else:
            samples = samples.astype(np.float32)

    return AudioStream(samples=samples, sample_rate=sample_rate)
