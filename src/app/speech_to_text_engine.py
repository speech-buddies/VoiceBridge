# src/app/speech_to_text_engine.py

from __future__ import annotations
from typing import List, Optional, Protocol

from src.models.audio_data import AudioStream, Transcript, AsrConfig
from src.app.whisper_tuned_model import WhisperLoraAsrModel
from src.utils.exceptions import InitializationError, ProcessingError, ModelValidationError


# Optional protocols for external dependencies
class VadInterface(Protocol):
    def detect_speech(self, audio: AudioStream) -> bool: ...


class NoiseFilterInterface(Protocol):
    def filter(self, audio: AudioStream) -> AudioStream: ...


class PersonalizationStore(Protocol):
    def get_profile(self) -> Optional[dict]: ...


class SpeechToTextEngine:
    """Speech-to-Text engine with optional VAD, noise filtering, and personalization."""

    EXPECTED_SAMPLE_RATE = 16_000
    EXPECTED_FORMAT = "PCM_16K_MONO"

    def __init__(
        self,
        config: AsrConfig,
        model_checkpoint: str,
        device: str = "cpu",
        vad: Optional[VadInterface] = None,
        noise_filter: Optional[NoiseFilterInterface] = None,
        personalization_store: Optional[PersonalizationStore] = None,
    ) -> None:
        self.config = config
        self.audio_buffer: List[AudioStream] = []

        self._vad = vad
        self._noise_filter = noise_filter
        self._personalization_store = personalization_store

        try:
            self.model: Optional[WhisperLoraAsrModel] = WhisperLoraAsrModel(
                model_checkpoint, device=device
            )
        except Exception as exc:
            raise InitializationError(f"Failed to initialize ASR model: {exc}") from exc

    def validateAudioFormat(self, audio: AudioStream) -> bool:
        """Check if audio matches expected rate/format."""
        is_rate_ok = audio.sample_rate == self.EXPECTED_SAMPLE_RATE
        is_format_ok = True  # format implied by mono float32; refine later if needed
        return is_rate_ok and is_format_ok

    def validateModelReady(self) -> bool:
        """Ensure ASR model is loaded and valid."""
        if self.model is None or not self.model.is_valid():
            raise ModelValidationError("ASR model not ready")
        return True

    def reset(self) -> None:
        """Clear audio buffer and reset model state."""
        self.audio_buffer.clear()
        if self.model:
            try:
                self.model.reset()
            except Exception:
                pass  # Whisper model is effectively stateless

    def processAudio(self, audio: AudioStream) -> Transcript:
        """Convert audio to transcript, applying preprocessing and personalization."""
        if not self.validateAudioFormat(audio):
            raise ProcessingError("Invalid audio format")

        self.audio_buffer.append(audio)

        processed_audio = audio
        if self._noise_filter:
            try:
                processed_audio = self._noise_filter.filter(audio)
            except Exception as exc:
                raise ProcessingError(f"Noise filtering failed: {exc}") from exc

        if self._vad:
            try:
                has_speech = self._vad.detect_speech(processed_audio)
            except Exception as exc:
                raise ProcessingError(f"VAD failed: {exc}") from exc
            if not has_speech:
                return Transcript(text="", confidence=0.0, metadata={"reason": "no_speech"})

        self.validateModelReady()

        try:
            features = self.model.extract_features(processed_audio)  # type: ignore[union-attr]
            transcript = self.model.decode(features)                 # type: ignore[union-attr]
        except Exception as exc:
            raise ProcessingError(f"Failed to process audio: {exc}") from exc

        return self._apply_personalization(transcript)

    def _apply_personalization(self, transcript: Transcript) -> Transcript:
        """Adjust transcript confidence and metadata using user profile."""
        if not self._personalization_store:
            return transcript

        profile = self._personalization_store.get_profile()
        if not profile:
            return transcript

        boost = float(profile.get("confidence_boost", 0.0))
        new_conf = max(0.0, min(1.0, transcript.confidence + boost))

        metadata = dict(transcript.metadata or {})
        if "id" in profile:
            metadata["profile_id"] = profile["id"]

        return Transcript(text=transcript.text, confidence=new_conf, metadata=metadata)
