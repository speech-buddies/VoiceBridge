"""
Real-time Audio Capture Module with Voice Activity Detection for Whisper ASR.

Responsibilities:
- Captures microphone input as a continuous audio stream (M15 in the MG/MIS).
- Detects speech boundaries using WebRTC VAD combined with an optional energy threshold.
- Buffers audio before and during speech, then emits complete recordings to a callback.
- Enforces configurable silence and maximum-duration limits to end recordings automatically.
"""

import numpy as np
import sounddevice as sd
import webrtcvad
from collections import deque
from threading import Thread, Event, Lock
from typing import Optional, Callable, Dict, Any
import time

from utils.logger import get_logger
logger = get_logger("AudioCapture")


class AudioConfig:
    """Holds all tunable parameters for audio capture and VAD behaviour.

    Derives chunk counts and buffer sizes from the human-readable duration
    parameters so the rest of the module works in uniform chunk units.
    """

    def __init__(
        self,
        sample_rate: int = 16000,            # PCM sample rate expected by WebRTC VAD and Whisper
        channels: int = 1,                   # Mono capture — required by WebRTC VAD
        chunk_duration_ms: int = 30,         # VAD frame size; must be 10, 20, or 30 ms
        vad_aggressiveness: int = 3,         # WebRTC VAD sensitivity (0 = least, 3 = most aggressive)
        pre_buffer_duration_ms: int = 300,   # Audio retained before speech onset to avoid clipping
        silence_duration_ms: int = 1500,     # Consecutive silence needed to end a recording
        max_recording_duration_s: int = 30,  # Hard cap on a single recording
        energy_threshold: Optional[float] = 800,  # RMS energy floor used alongside VAD
        device: Optional[int] = None         # sounddevice device index; None = system default
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

        # Derived chunk counts — computed once so no floating-point division happens at runtime
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.pre_buffer_chunks = int(pre_buffer_duration_ms / chunk_duration_ms)
        self.silence_chunks = int(silence_duration_ms / chunk_duration_ms)
        self.max_recording_chunks = int(max_recording_duration_s * 1000 / chunk_duration_ms)

    def validate(self) -> bool:
        """Confirms the configuration is compatible with WebRTC VAD constraints."""
        valid_rates = [8000, 16000, 32000, 48000]
        if self.sample_rate not in valid_rates:
            logger.warning(f"Sample rate {self.sample_rate} not supported by WebRTC VAD. Use one of {valid_rates}")
            return False

        if self.chunk_duration_ms not in [10, 20, 30]:
            logger.warning(f"Chunk duration {self.chunk_duration_ms}ms not supported by WebRTC VAD. Use 10, 20, or 30ms")
            return False

        return True


class AudioState:
    """Enumerates the states of the capture state machine.

    Transitions:
      IDLE -> LISTENING (on start)
      LISTENING -> SPEECH_DETECTED (on first speech frame)
      SPEECH_DETECTED -> RECORDING (on next chunk)
      RECORDING -> PROCESSING (on silence / max-duration limit)
      PROCESSING -> LISTENING (after callback completes)
      Any -> STOPPED (on stop)
      STOPPED -> IDLE (cleanup)
    """
    IDLE = "idle"
    LISTENING = "listening"
    SPEECH_DETECTED = "speech_detected"
    RECORDING = "recording"
    PROCESSING = "processing"
    STOPPED = "stopped"


class RealtimeAudioCapture:
    """Manages the full lifecycle of real-time audio capture with VAD.

    Continuously reads from the microphone, classifies each chunk as speech
    or silence, and emits complete utterances to the registered callback.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        on_audio_ready: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None
    ):
        self.config = config or AudioConfig()
        if not self.config.validate():
            raise ValueError("Invalid audio configuration")

        # Callback invoked with (float32 audio array, metadata dict) after each utterance
        self.on_audio_ready = on_audio_ready

        self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)

        self.state = AudioState.IDLE
        self.state_lock = Lock()

        # Ring buffer holding recent chunks to prepend to recordings (avoids clipped speech onset)
        self.pre_buffer = deque(maxlen=self.config.pre_buffer_chunks)

        self.recording_buffer = []

        self.silence_counter = 0
        self.recording_chunks_count = 0

        self.stop_event = Event()
        self.capture_thread: Optional[Thread] = None

        # Diagnostic counters exposed via get_stats()
        self.stats = {
            'total_recordings': 0,
            'total_audio_seconds': 0.0,
            'vad_errors': 0,
            'stream_errors': 0
        }

        logger.info(f"AudioCapture initialized: {self.config.sample_rate}Hz, VAD level {self.config.vad_aggressiveness}")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, new_state: str):
        """Thread-safe state transition with debug logging."""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            if old_state != new_state:
                logger.debug(f"State transition: {old_state} -> {new_state}")

    def _get_state(self) -> str:
        """Thread-safe state read."""
        with self.state_lock:
            return self.state

    # ------------------------------------------------------------------
    # Speech detection
    # ------------------------------------------------------------------

    def _is_speech(self, audio_chunk: bytes) -> bool:
        """Classifies a raw PCM chunk as speech or silence.

        Uses WebRTC VAD as the primary detector; optionally augments with an
        RMS energy check so low-energy voiced sounds are not missed.
        """
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

    # ------------------------------------------------------------------
    # Chunk processing — state machine dispatch
    # ------------------------------------------------------------------

    def _process_audio_chunk(self, audio_chunk: bytes):
        """Routes each incoming PCM chunk through the capture state machine."""
        current_state = self._get_state()
        is_speech = self._is_speech(audio_chunk)

        if current_state == AudioState.LISTENING:
            # Maintain the pre-roll buffer; start a recording when speech is detected
            self.pre_buffer.append(audio_chunk)
            if is_speech:
                logger.info("Speech detected, starting recording")
                self._set_state(AudioState.SPEECH_DETECTED)
                self._start_recording()

        elif current_state == AudioState.SPEECH_DETECTED:
            # Transition to RECORDING on the very next chunk after detection
            self.recording_buffer.append(audio_chunk)
            self.recording_chunks_count += 1
            self._set_state(AudioState.RECORDING)
            self.silence_counter = 0

        elif current_state == AudioState.RECORDING:
            # Accumulate chunks; end the recording on sustained silence or max duration
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

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def _start_recording(self):
        """Seeds the recording buffer with pre-buffered audio to preserve speech onset."""
        self.recording_buffer = list(self.pre_buffer)
        self.recording_chunks_count = len(self.recording_buffer)
        self.silence_counter = 0

    def _end_recording(self):
        """Finalizes the recording, converts to float32, and fires the audio callback."""
        self._set_state(AudioState.PROCESSING)

        # Concatenate raw bytes, convert int16 PCM to normalized float32 expected by Whisper
        audio_bytes = b''.join(self.recording_buffer)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        duration_s = len(audio_float32) / self.config.sample_rate

        # Metadata surfaced to the callback for downstream logging and processing
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

        # Reset per-utterance state and resume listening
        self.recording_buffer = []
        self.recording_chunks_count = 0
        self.silence_counter = 0
        self._set_state(AudioState.LISTENING)

    # ------------------------------------------------------------------
    # sounddevice stream callback
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        """Invoked by sounddevice for each captured audio block.

        Converts the float32 block to int16 bytes and forwards it to the
        state machine. Stream errors are counted but do not halt capture.
        """
        if status:
            logger.warning(f"Audio stream status: {status}")
            self.stats['stream_errors'] += 1

        if self._get_state() == AudioState.STOPPED:
            return

        # sounddevice delivers float32; VAD and int16 arithmetic require int16 bytes
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        self._process_audio_chunk(audio_bytes)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        """Opens the microphone stream and begins continuous capture."""
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
        """Signals the capture loop to halt and closes the audio stream."""
        logger.info("Stopping audio capture...")
        self._set_state(AudioState.STOPPED)
        self.stop_event.set()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        self._set_state(AudioState.IDLE)
        logger.info("Audio capture stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Returns a snapshot of runtime diagnostic counters."""
        return self.stats.copy()

    @staticmethod
    def list_audio_devices():
        """Enumerates available system input devices for configuration purposes."""
        devices = sd.query_devices()
        input_devices = []
        print("\nAvailable Audio Input Devices:")
        print("-" * 80)
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((idx, device))
                print(f"[{idx}] {device['name']}")
        return input_devices