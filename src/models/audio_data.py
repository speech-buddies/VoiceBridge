from dataclasses import dataclass
import numpy as np

@dataclass
class AudioStream:
    samples: np.ndarray   # float32 audio samples
    sample_rate: int      # sample rate in Hz

@dataclass
class Transcript:
    text: str
    confidence: float
    metadata: dict

@dataclass
class AsrConfig:
    sample_rate: int
    frame_size: int
