"""
Unit tests for SpeechToTextEngine.
Run with: pytest test/test_app/test_speech_to_text_engine.py -v --cov=app.speech_to_text_engine --cov-report=term-missing
"""

import asyncio
import os
import sys
import types
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Stub out all heavy / unavailable dependencies before any project import
# ---------------------------------------------------------------------------

def _stub_leaf(dotted_name: str, **attrs):
    """
    Register a stub module at `dotted_name` only if not already present.
    Never touches parent package entries so real packages aren't shadowed.
    """
    if dotted_name not in sys.modules:
        mod = types.ModuleType(dotted_name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[dotted_name] = mod
    else:
        # Module exists — just patch in any missing attrs
        mod = sys.modules[dotted_name]
        for k, v in attrs.items():
            if not hasattr(mod, k):
                setattr(mod, k, v)
    return sys.modules[dotted_name]


# --- models.audio_data ---
class _AudioStream:
    def __init__(self, samples=None, sample_rate=16000, format=None):
        self.samples = samples if samples is not None else np.zeros(16000, dtype=np.float32)
        self.sample_rate = sample_rate
        self.format = format

class _Transcript:
    def __init__(self, text="", metadata=None):
        self.text = text
        self.metadata = metadata or {}

class _AsrConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

_stub_leaf("models", **{})
_stub_leaf("models.audio_data",
           AudioStream=_AudioStream, Transcript=_Transcript, AsrConfig=_AsrConfig)

# --- app.whisper_tuned_model (leaf only — never register plain 'app') ---
_MockWhisperClass = MagicMock()
_stub_leaf("app.whisper_tuned_model", WhisperLoraAsrModel=_MockWhisperClass)

# --- utils.exceptions ---
class InitializationError(Exception): pass
class ProcessingError(Exception): pass
class ModelValidationError(Exception): pass

_stub_leaf("utils", **{})
_stub_leaf("utils.exceptions",
           InitializationError=InitializationError,
           ProcessingError=ProcessingError,
           ModelValidationError=ModelValidationError)

# --- utils.logger ---
_stub_logger = MagicMock()
_stub_leaf("utils.logger", get_logger=MagicMock(return_value=_stub_logger))

# --- src.input.wav_loader ---
_stub_leaf("src", **{})
_stub_leaf("src.input", **{})
_stub_leaf("src.input.wav_loader", wav_to_audiostream=MagicMock())


# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------
from app.speech_to_text_engine import SpeechToTextEngine  # noqa: E402
from models.audio_data import AudioStream, Transcript, AsrConfig
from utils.exceptions import InitializationError, ProcessingError, ModelValidationError


# ===========================================================================
# Helpers
# ===========================================================================

SAMPLE_RATE = 16_000


def _make_engine(
    noise_filter=None,
    personalization_store=None,
    model_checkpoint="./models/adapters/whisper_lora_epoch1.pt",
    device="cpu",
    config=None,
) -> SpeechToTextEngine:
    """Return a fresh engine with a fully mocked WhisperLoraAsrModel."""
    mock_model = MagicMock()
    mock_model.is_valid.return_value = True
    mock_model.extract_features.return_value = MagicMock()
    mock_model.decode.return_value = _Transcript(text="hello world")

    with patch("app.speech_to_text_engine.WhisperLoraAsrModel", return_value=mock_model):
        engine = SpeechToTextEngine(
            config=config,
            model_checkpoint=model_checkpoint,
            device=device,
            noise_filter=noise_filter,
            personalization_store=personalization_store,
        )
    return engine


def _make_audio_bytes(n_samples: int = SAMPLE_RATE, amplitude: int = 1000) -> bytes:
    """Generate raw PCM int16 bytes."""
    return (np.ones(n_samples, dtype=np.int16) * amplitude).tobytes()


def _make_audio_stream(sample_rate: int = SAMPLE_RATE) -> AudioStream:
    return _AudioStream(
        samples=np.zeros(SAMPLE_RATE, dtype=np.float32),
        sample_rate=sample_rate,
    )


# ===========================================================================
# __init__
# ===========================================================================

class TestInit:

    def test_model_initialized(self):
        engine = _make_engine()
        assert engine.model is not None

    def test_audio_buffer_empty_on_init(self):
        engine = _make_engine()
        assert engine.audio_buffer == []

    def test_config_stored(self):
        cfg = _AsrConfig(language="en")
        engine = _make_engine(config=cfg)
        assert engine.config is cfg

    def test_noise_filter_stored(self):
        nf = MagicMock()
        engine = _make_engine(noise_filter=nf)
        assert engine._noise_filter is nf

    def test_personalization_store_stored(self):
        ps = MagicMock()
        engine = _make_engine(personalization_store=ps)
        assert engine._personalization_store is ps

    def test_model_init_failure_raises_initialization_error(self):
        with patch(
            "app.speech_to_text_engine.WhisperLoraAsrModel",
            side_effect=RuntimeError("checkpoint missing"),
        ):
            with pytest.raises(InitializationError, match="Failed to initialize ASR model"):
                SpeechToTextEngine()

    def test_default_noise_filter_is_none(self):
        engine = _make_engine()
        assert engine._noise_filter is None

    def test_default_personalization_store_is_none(self):
        engine = _make_engine()
        assert engine._personalization_store is None


# ===========================================================================
# validateAudioFormat
# ===========================================================================

class TestValidateAudioFormat:

    def test_valid_audio_returns_true(self):
        engine = _make_engine()
        audio = _make_audio_stream(sample_rate=SAMPLE_RATE)
        assert engine.validateAudioFormat(audio) is True

    def test_wrong_sample_rate_returns_false(self):
        engine = _make_engine()
        audio = _make_audio_stream(sample_rate=8000)
        assert engine.validateAudioFormat(audio) is False

    def test_correct_sample_rate_boundary(self):
        engine = _make_engine()
        audio = _make_audio_stream(sample_rate=16000)
        assert engine.validateAudioFormat(audio) is True

    def test_44100_sample_rate_returns_false(self):
        engine = _make_engine()
        audio = _make_audio_stream(sample_rate=44100)
        assert engine.validateAudioFormat(audio) is False


# ===========================================================================
# validateModelReady
# ===========================================================================

class TestValidateModelReady:

    def test_valid_model_returns_true(self):
        engine = _make_engine()
        engine.model.is_valid.return_value = True
        assert engine.validateModelReady() is True

    def test_invalid_model_raises(self):
        engine = _make_engine()
        engine.model.is_valid.return_value = False
        with pytest.raises(ModelValidationError, match="ASR model not ready"):
            engine.validateModelReady()

    def test_none_model_raises(self):
        engine = _make_engine()
        engine.model = None
        with pytest.raises(ModelValidationError, match="ASR model not ready"):
            engine.validateModelReady()


# ===========================================================================
# reset
# ===========================================================================

class TestReset:

    def test_clears_audio_buffer(self):
        engine = _make_engine()
        engine.audio_buffer.append(_make_audio_stream())
        engine.reset()
        assert engine.audio_buffer == []

    def test_calls_model_reset(self):
        engine = _make_engine()
        engine.reset()
        engine.model.reset.assert_called_once()

    def test_model_reset_exception_is_swallowed(self):
        engine = _make_engine()
        engine.model.reset.side_effect = Exception("stateless")
        engine.reset()  # must not raise
        assert engine.audio_buffer == []

    def test_reset_with_none_model_does_not_raise(self):
        engine = _make_engine()
        engine.model = None
        engine.reset()  # must not raise

    def test_reset_idempotent(self):
        engine = _make_engine()
        engine.audio_buffer.append(_make_audio_stream())
        engine.reset()
        engine.reset()
        assert engine.audio_buffer == []


# ===========================================================================
# _bytes_to_audio_stream
# ===========================================================================

class TestBytesToAudioStream:

    def test_returns_audio_stream(self):
        engine = _make_engine()
        result = engine._bytes_to_audio_stream(_make_audio_bytes())
        assert isinstance(result, _AudioStream)

    def test_sample_rate_is_16k(self):
        engine = _make_engine()
        result = engine._bytes_to_audio_stream(_make_audio_bytes())
        assert result.sample_rate == SAMPLE_RATE

    def test_samples_are_float32(self):
        engine = _make_engine()
        result = engine._bytes_to_audio_stream(_make_audio_bytes())
        assert result.samples.dtype == np.float32

    def test_samples_normalized_in_range(self):
        engine = _make_engine()
        # Maximum int16 amplitude
        loud = (np.ones(100, dtype=np.int16) * 32767).tobytes()
        result = engine._bytes_to_audio_stream(loud)
        assert np.all(result.samples <= 1.0)
        assert np.all(result.samples >= -1.0)

    def test_sample_count_matches_input(self):
        engine = _make_engine()
        n = 8000
        audio_bytes = _make_audio_bytes(n_samples=n)
        result = engine._bytes_to_audio_stream(audio_bytes)
        assert len(result.samples) == n

    def test_zero_bytes_produce_zero_samples(self):
        engine = _make_engine()
        silent = np.zeros(100, dtype=np.int16).tobytes()
        result = engine._bytes_to_audio_stream(silent)
        assert np.all(result.samples == 0.0)

    def test_negative_amplitude_preserved(self):
        engine = _make_engine()
        negative = (np.ones(100, dtype=np.int16) * -1000).tobytes()
        result = engine._bytes_to_audio_stream(negative)
        assert np.all(result.samples < 0)


# ===========================================================================
# transcribe (async)
# ===========================================================================

class TestTranscribe:

    def setup_method(self):
        self.engine = _make_engine()
        self.engine.model.decode.return_value = _Transcript(text="hello world")

    @pytest.mark.asyncio
    async def test_returns_transcript_text(self):
        result = await self.engine.transcribe(_make_audio_bytes())
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_calls_extract_features(self):
        await self.engine.transcribe(_make_audio_bytes())
        self.engine.model.extract_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_decode(self):
        await self.engine.transcribe(_make_audio_bytes())
        self.engine.model.decode.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_sample_rate_raises_runtime_error(self):
        engine = _make_engine()
        # Patch _bytes_to_audio_stream to return wrong sample rate
        bad_stream = _make_audio_stream(sample_rate=8000)
        engine._bytes_to_audio_stream = MagicMock(return_value=bad_stream)
        with pytest.raises(RuntimeError, match="Failed to transcribe"):
            await engine.transcribe(_make_audio_bytes())

    @pytest.mark.asyncio
    async def test_invalid_model_raises_runtime_error(self):
        self.engine.model.is_valid.return_value = False
        with pytest.raises(RuntimeError, match="Failed to transcribe"):
            await self.engine.transcribe(_make_audio_bytes())

    @pytest.mark.asyncio
    async def test_noise_filter_applied_when_present(self):
        nf = MagicMock()
        nf.filter.return_value = _make_audio_stream()
        engine = _make_engine(noise_filter=nf)
        engine.model.decode.return_value = _Transcript(text="filtered")
        await engine.transcribe(_make_audio_bytes())
        nf.filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_noise_filter_exception_raises_runtime_error(self):
        nf = MagicMock()
        nf.filter.side_effect = Exception("filter exploded")
        engine = _make_engine(noise_filter=nf)
        with pytest.raises(RuntimeError, match="Failed to transcribe"):
            await engine.transcribe(_make_audio_bytes())

    @pytest.mark.asyncio
    async def test_no_noise_filter_skips_filtering(self):
        # No noise filter — model still called directly
        await self.engine.transcribe(_make_audio_bytes())
        self.engine.model.extract_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcript_obj_with_text_attr_used(self):
        self.engine.model.decode.return_value = _Transcript(text="specific text")
        result = await self.engine.transcribe(_make_audio_bytes())
        assert result == "specific text"

    @pytest.mark.asyncio
    async def test_transcript_obj_without_text_attr_stringified(self):
        self.engine.model.decode.return_value = "raw string transcript"
        result = await self.engine.transcribe(_make_audio_bytes())
        assert result == "raw string transcript"

    @pytest.mark.asyncio
    async def test_extract_features_failure_raises_runtime_error(self):
        self.engine.model.extract_features.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(RuntimeError, match="Failed to transcribe"):
            await self.engine.transcribe(_make_audio_bytes())

    @pytest.mark.asyncio
    async def test_personalization_applied_to_result(self):
        ps = MagicMock()
        ps.get_profile.return_value = {"id": "user-42"}
        engine = _make_engine(personalization_store=ps)
        # transcribe passes a plain string to _apply_personalization;
        # since str has no .metadata, it returns the string unchanged
        engine.model.decode.return_value = _Transcript(text="hello")
        engine._apply_personalization = MagicMock(return_value="hello")
        result = await engine.transcribe(_make_audio_bytes())
        engine._apply_personalization.assert_called_once_with("hello")


# ===========================================================================
# processAudio
# ===========================================================================

class TestProcessAudio:

    def test_invalid_format_raises_processing_error(self):
        engine = _make_engine()
        bad_audio = _make_audio_stream(sample_rate=8000)
        with pytest.raises(ProcessingError, match="Invalid audio format"):
            engine.processAudio(bad_audio)

    def test_valid_audio_appended_to_buffer(self):
        engine = _make_engine()
        audio = _make_audio_stream()
        engine.processAudio(audio)
        assert audio in engine.audio_buffer

    def test_multiple_audio_chunks_accumulated(self):
        engine = _make_engine()
        for _ in range(3):
            engine.processAudio(_make_audio_stream())
        assert len(engine.audio_buffer) == 3


# ===========================================================================
# _apply_personalization
# ===========================================================================

class TestApplyPersonalization:

    def test_no_store_returns_transcript_unchanged(self):
        engine = _make_engine()
        t = _Transcript(text="hello", metadata={"key": "val"})
        result = engine._apply_personalization(t)
        assert result is t

    def test_store_with_no_profile_returns_transcript_unchanged(self):
        ps = MagicMock()
        ps.get_profile.return_value = None
        engine = _make_engine(personalization_store=ps)
        t = _Transcript(text="hello")
        result = engine._apply_personalization(t)
        assert result.text == "hello"

    def test_profile_id_added_to_metadata(self):
        ps = MagicMock()
        ps.get_profile.return_value = {"id": "user-99"}
        engine = _make_engine(personalization_store=ps)
        t = _Transcript(text="hello", metadata={})
        result = engine._apply_personalization(t)
        assert result.metadata["profile_id"] == "user-99"

    def test_existing_metadata_preserved(self):
        ps = MagicMock()
        ps.get_profile.return_value = {"id": "user-99"}
        engine = _make_engine(personalization_store=ps)
        t = _Transcript(text="hello", metadata={"lang": "en"})
        result = engine._apply_personalization(t)
        assert result.metadata["lang"] == "en"
        assert result.metadata["profile_id"] == "user-99"

    def test_profile_without_id_does_not_add_profile_id(self):
        ps = MagicMock()
        ps.get_profile.return_value = {"name": "Alice"}  # no 'id' key
        engine = _make_engine(personalization_store=ps)
        t = _Transcript(text="hello", metadata={})
        result = engine._apply_personalization(t)
        assert "profile_id" not in result.metadata

    def test_none_metadata_handled(self):
        ps = MagicMock()
        ps.get_profile.return_value = {"id": "user-1"}
        engine = _make_engine(personalization_store=ps)
        t = _Transcript(text="hi", metadata=None)
        result = engine._apply_personalization(t)
        assert result.metadata["profile_id"] == "user-1"

    def test_returns_new_transcript_object(self):
        ps = MagicMock()
        ps.get_profile.return_value = {"id": "user-1"}
        engine = _make_engine(personalization_store=ps)
        t = _Transcript(text="hi", metadata={})
        result = engine._apply_personalization(t)
        assert result is not t


# ===========================================================================
# Integration: transcribe end-to-end with noise filter + personalization
# ===========================================================================

class TestTranscribeIntegration:

    @pytest.mark.asyncio
    async def test_noise_filter_then_personalization(self):
        nf = MagicMock()
        filtered_audio = _make_audio_stream()
        nf.filter.return_value = filtered_audio

        ps = MagicMock()
        ps.get_profile.return_value = {"id": "user-7"}

        engine = _make_engine(noise_filter=nf, personalization_store=ps)
        engine.model.decode.return_value = _Transcript(text="transcribed")
        # Intercept _apply_personalization so it handles the str it receives
        engine._apply_personalization = MagicMock(return_value="transcribed")

        result = await engine.transcribe(_make_audio_bytes())

        nf.filter.assert_called_once()
        engine._apply_personalization.assert_called_once_with("transcribed")

    @pytest.mark.asyncio
    async def test_full_pipeline_returns_string(self):
        engine = _make_engine()
        engine.model.decode.return_value = _Transcript(text="full pipeline result")
        result = await engine.transcribe(_make_audio_bytes())
        assert isinstance(result, str)
        assert result == "full pipeline result"