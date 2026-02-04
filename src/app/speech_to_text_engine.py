from __future__ import annotations
from typing import List, Optional, Protocol

from models.audio_data import AudioStream, Transcript, AsrConfig
from app.whisper_tuned_model import WhisperLoraAsrModel

from utils.exceptions import InitializationError, ProcessingError, ModelValidationError
from utils import logger

logger = logger.get_logger("SpeechToTextEngine")

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
        noise_filter: Optional[NoiseFilterInterface] = None,
        personalization_store: Optional[PersonalizationStore] = None,
    ) -> None:
        self.config = config
        self.audio_buffer: List[AudioStream] = []

        self._noise_filter = noise_filter
        self._personalization_store = personalization_store

        try:
            self.model: Optional[WhisperLoraAsrModel] = WhisperLoraAsrModel(
                model_checkpoint=model_checkpoint,
                device=device,
                adapter_path=model_checkpoint
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

    def _bytes_to_audio_stream(self, audio_data: bytes):
        """
        Convert raw audio bytes to AudioStream object.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit mono at 16kHz)
            
        Returns:
            AudioStream object ready for model processing
        """
        import numpy as np
        from models.audio_data import AudioStream
        
        # Convert bytes to numpy array of int16
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        # Convert to float32 normalized to [-1, 1]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        # Create AudioStream object
        audio_stream = AudioStream(
            data=audio_float32,
            sample_rate=self.EXPECTED_SAMPLE_RATE,
            format=self.EXPECTED_FORMAT
        )
        
        return audio_stream

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio bytes to text.This is the main method called by the server.
        
        Args:
            audio_data: Raw audio bytes from VAD (PCM 16kHz mono, 16-bit)
            
        Returns:
            Transcript text string
            
        Raises:
            ValueError: If audio format is invalid
            RuntimeError: If transcription fails
        """
        
        try:
            # Convert bytes to AudioStream format expected by model
            audio_stream = self._bytes_to_audio_stream(audio_data)

            # Validate audio format
            if not self.validateAudioFormat(audio_data):
                raise ProcessingError("Invalid audio format")
            
            self.validateModelReady()
            
            if self._noise_filter:
                try:
                    processed_audio = self._noise_filter.filter(audio_stream)
                except Exception as exc:
                    raise ProcessingError(f"Noise filtering failed: {exc}") from exc
            # Extract features using Whisper model
            features = self.model.extract_features(audio_stream)
            
            # Decode features to get transcript
            transcript_obj = self.model.decode(features)
            
            # Extract text from transcript object
            if hasattr(transcript_obj, 'text'):
                transcript_text = transcript_obj.text
            else:
                # Fallback if transcript_obj is already a string
                transcript_text = str(transcript_obj)
            
            return self._apply_personalization(transcript_text)
            
        except Exception as exc:
            logger.error(f"Transcription failed: {exc}", exc_info=True)
            raise RuntimeError(f"Failed to transcribe audio: {exc}") from exc


    def processAudio(self, audio: AudioStream) -> Transcript:
        """Convert audio to transcript, applying preprocessing and personalization."""
        if not self.validateAudioFormat(audio):
            raise ProcessingError("Invalid audio format")

        self.audio_buffer.append(audio)

        processed_audio = audio
        

    def _apply_personalization(self, transcript: Transcript) -> Transcript:
        """Optionally adjust transcript metadata using user profile."""
        if not self._personalization_store:
            return transcript

        profile = self._personalization_store.get_profile()
        if not profile:
            return transcript

        metadata = dict(transcript.metadata or {})
        if "id" in profile:
            metadata["profile_id"] = profile["id"]

        return Transcript(text=transcript.text, metadata=metadata)

    def transcribe_folder(self, folder_path: str) -> Transcript:
        """Transcribe all WAV files in a folder, ordered by filename, and combine results."""
        import os
        from src.input.wav_loader import wav_to_audiostream

        wav_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
        )
        combined_text = []
        metadatas = []

        for fname in wav_files:
            fpath = os.path.join(folder_path, fname)
            audio = wav_to_audiostream(fpath)
            transcript = self.processAudio(audio)
            combined_text.append(transcript.text)
            metadatas.append(transcript.metadata)

        final_text = " ".join(combined_text).strip()
        combined_metadata = {"chunks": len(wav_files), "sources": wav_files}
        if metadatas:
            combined_metadata.update(metadatas[0])

        return Transcript(text=final_text, metadata=combined_metadata)
