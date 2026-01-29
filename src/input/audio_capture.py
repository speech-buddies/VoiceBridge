"""
Real-time Audio Capture Module with Voice Activity Detection for Whisper ASR
Supports continuous listening, automatic speech detection, and audio buffering.
"""

import numpy as np
import sounddevice as sd
import webrtcvad
from collections import deque
from threading import Thread, Event, Lock
from typing import Optional, Callable, Dict, Any
import time

# --- MODIFIED: Import local logger utility ---
from utils.logger import get_logger

# Initialize the logger for this specific module
logger = get_logger("AudioCapture")
# ---------------------------------------------

class AudioConfig:
    """Configuration for audio capture and processing"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        vad_aggressiveness: int = 3,
        pre_buffer_duration_ms: int = 300,
        silence_duration_ms: int = 1500,
        max_recording_duration_s: int = 30,
        energy_threshold: Optional[float] = None,
        device: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.vad_aggressiveness = vad_aggressiveness
        self.pre_buffer_duration_ms = pre_buffer_duration_ms
        self.silence_duration_ms = silence_duration_ms
        self.max_recording_duration_s = max_recording_duration_s
        self.energy_threshold = energy_threshold
        self.device = device
        
        # Derived parameters
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.pre_buffer_chunks = int(pre_buffer_duration_ms / chunk_duration_ms)
        self.silence_chunks = int(silence_duration_ms / chunk_duration_ms)
        self.max_recording_chunks = int(max_recording_duration_s * 1000 / chunk_duration_ms)
        
    def validate(self) -> bool:
        """Validate configuration parameters"""
        valid_rates = [8000, 16000, 32000, 48000]
        if self.sample_rate not in valid_rates:
            logger.warning(f"Sample rate {self.sample_rate} not supported by WebRTC VAD. Use one of {valid_rates}")
            return False
            
        if self.chunk_duration_ms not in [10, 20, 30]:
            logger.warning(f"Chunk duration {self.chunk_duration_ms}ms not supported by WebRTC VAD. Use 10, 20, or 30ms")
            return False
            
        return True


class AudioState:
    IDLE = "idle"
    LISTENING = "listening"
    SPEECH_DETECTED = "speech_detected"
    RECORDING = "recording"
    PROCESSING = "processing"
    STOPPED = "stopped"


class RealtimeAudioCapture:
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        on_audio_ready: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None
    ):
        self.config = config or AudioConfig()
        if not self.config.validate():
            raise ValueError("Invalid audio configuration")
            
        self.on_audio_ready = on_audio_ready
        self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)
        self.state = AudioState.IDLE
        self.state_lock = Lock()
        self.pre_buffer = deque(maxlen=self.config.pre_buffer_chunks)
        self.recording_buffer = []
        self.silence_counter = 0
        self.recording_chunks_count = 0
        self.stop_event = Event()
        self.capture_thread: Optional[Thread] = None
        
        self.stats = {
            'total_recordings': 0,
            'total_audio_seconds': 0.0,
            'vad_errors': 0,
            'stream_errors': 0
        }
        
        logger.info(f"AudioCapture initialized: {self.config.sample_rate}Hz, VAD level {self.config.vad_aggressiveness}")
    
    def _set_state(self, new_state: str):
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            if old_state != new_state:
                # These will now go into your 'app.log' automatically
                logger.debug(f"State transition: {old_state} -> {new_state}")
    
    def _get_state(self) -> str:
        with self.state_lock:
            return self.state
    
    def _is_speech(self, audio_chunk: bytes) -> bool:
        try:
            is_speech_vad = self.vad.is_speech(audio_chunk, self.config.sample_rate)
            if self.config.energy_threshold is not None:
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                is_speech_energy = energy > self.config.energy_threshold
                return is_speech_vad or is_speech_energy
            return is_speech_vad
        except Exception as e:
            self.stats['vad_errors'] += 1
            logger.error(f"VAD error: {e}")
            return False
    
    def _process_audio_chunk(self, audio_chunk: bytes):
        current_state = self._get_state()
        is_speech = self._is_speech(audio_chunk)
        
        if current_state == AudioState.LISTENING:
            self.pre_buffer.append(audio_chunk)
            if is_speech:
                logger.info("Speech detected, starting recording")
                self._set_state(AudioState.SPEECH_DETECTED)
                self._start_recording()
                
        elif current_state == AudioState.SPEECH_DETECTED:
            self.recording_buffer.append(audio_chunk)
            self.recording_chunks_count += 1
            self._set_state(AudioState.RECORDING)
            self.silence_counter = 0
            
        elif current_state == AudioState.RECORDING:
            self.recording_buffer.append(audio_chunk)
            self.recording_chunks_count += 1
            if is_speech:
                self.silence_counter = 0
            else:
                self.silence_counter += 1
            
            if self.silence_counter >= self.config.silence_chunks:
                logger.info(f"Silence detected ({self.config.silence_duration_ms}ms), ending recording")
                self._end_recording()
            elif self.recording_chunks_count >= self.config.max_recording_chunks:
                logger.info(f"Max duration reached ({self.config.max_recording_duration_s}s), ending recording")
                self._end_recording()
    
    def _start_recording(self):
        self.recording_buffer = list(self.pre_buffer)
        self.recording_chunks_count = len(self.recording_buffer)
        self.silence_counter = 0
    
    def _end_recording(self):
        self._set_state(AudioState.PROCESSING)
        audio_bytes = b''.join(self.recording_buffer)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        duration_s = len(audio_float32) / self.config.sample_rate
        
        metadata = {
            'duration_s': duration_s,
            'sample_rate': self.config.sample_rate,
            'num_samples': len(audio_float32),
            'timestamp': time.time()
        }
        
        logger.info(f"Recording complete: {duration_s:.2f}s, {len(audio_float32)} samples")
        self.stats['total_recordings'] += 1
        self.stats['total_audio_seconds'] += duration_s
        
        if self.on_audio_ready:
            try:
                self.on_audio_ready(audio_float32, metadata)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        
        self.recording_buffer = []
        self.recording_chunks_count = 0
        self.silence_counter = 0
        self._set_state(AudioState.LISTENING)
    
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio stream status: {status}")
            self.stats['stream_errors'] += 1
        
        if self._get_state() == AudioState.STOPPED:
            return
        
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        self._process_audio_chunk(audio_bytes)
    
    def start(self):
        if self._get_state() != AudioState.IDLE:
            logger.warning("Audio capture already running")
            return
        
        self.stop_event.clear()
        self._set_state(AudioState.LISTENING)
        
        try:
            self.stream = sd.InputStream(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.chunk_size,
                dtype='float32',
                callback=self._audio_callback
            )
            self.stream.start()
            logger.info("Audio capture started")
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self._set_state(AudioState.IDLE)
            raise
    
    def stop(self):
        logger.info("Stopping audio capture...")
        self._set_state(AudioState.STOPPED)
        self.stop_event.set()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self._set_state(AudioState.IDLE)
        logger.info("Audio capture stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    @staticmethod
    def list_audio_devices():
        devices = sd.query_devices()
        input_devices = []
        print("\nAvailable Audio Input Devices:")
        print("-" * 80)
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((idx, device))
                print(f"[{idx}] {device['name']}")
        return input_devices


if __name__ == "__main__":
    RealtimeAudioCapture.list_audio_devices()
    
    def on_audio_captured(audio_array: np.ndarray, metadata: Dict[str, Any]):
        # Now using standard print for the user interface, 
        # but the module internals use the logger.
        print(f"\n🎤 Audio captured! {metadata['duration_s']:.2f}s")
    
    config = AudioConfig(sample_rate=16000, vad_aggressiveness=3)
    capture = RealtimeAudioCapture(config=config, on_audio_ready=on_audio_captured)
    
    try:
        print("\n🎙️ Starting audio capture (Check app.log for debug details)...")
        capture.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        capture.stop()