"""
Unit tests for the Real-time Audio Capture Module.
Run with: pytest test_audio_capture.py -v --cov=audio_capture --cov-report=term-missing
"""

import numpy as np
import pytest
import time
from collections import deque
from threading import Event, Lock
from unittest.mock import MagicMock, patch, PropertyMock, call

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without the real dependencies
# ---------------------------------------------------------------------------

import sys
import types

# --- stub: utils.logger ---
utils_pkg = types.ModuleType("utils")
logger_mod = types.ModuleType("utils.logger")
_stub_logger = MagicMock()
logger_mod.get_logger = MagicMock(return_value=_stub_logger)
sys.modules.setdefault("utils", utils_pkg)
sys.modules.setdefault("utils.logger", logger_mod)

# --- stub: sounddevice ---
sd_mod = types.ModuleType("sounddevice")
sd_mod.query_devices = MagicMock(return_value=[])
_FakeInputStream = MagicMock
sd_mod.InputStream = _FakeInputStream
sys.modules.setdefault("sounddevice", sd_mod)

# --- stub: webrtcvad ---
webrtcvad_mod = types.ModuleType("webrtcvad")
webrtcvad_mod.Vad = MagicMock()
sys.modules.setdefault("webrtcvad", webrtcvad_mod)

# Now import the module under test
import importlib, audio_capture as _ac_module
from audio_capture import (

    AudioConfig,
    AudioState,
    RealtimeAudioCapture,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_capture(config=None, callback=None):
    """Return a fresh RealtimeAudioCapture with a mocked VAD."""
    cap = RealtimeAudioCapture(config=config, on_audio_ready=callback)
    cap.vad = MagicMock()
    cap.vad.is_speech.return_value = False
    return cap


def _raw_chunk(capture: RealtimeAudioCapture, is_speech: bool = False) -> bytes:
    """Generate a valid int16 PCM byte chunk matching the capture config."""
    capture.vad.is_speech.return_value = is_speech
    n = capture.config.chunk_size
    return (np.zeros(n, dtype=np.int16)).tobytes()


# ===========================================================================
# AudioConfig tests
# ===========================================================================

class TestAudioConfig:

    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.chunk_duration_ms == 30
        assert cfg.vad_aggressiveness == 3

    def test_derived_chunk_size(self):
        cfg = AudioConfig(sample_rate=16000, chunk_duration_ms=30)
        assert cfg.chunk_size == 480          # 16000 * 30 / 1000

    def test_derived_pre_buffer_chunks(self):
        cfg = AudioConfig(pre_buffer_duration_ms=300, chunk_duration_ms=30)
        assert cfg.pre_buffer_chunks == 10

    def test_derived_silence_chunks(self):
        cfg = AudioConfig(silence_duration_ms=1500, chunk_duration_ms=30)
        assert cfg.silence_chunks == 50

    def test_derived_max_recording_chunks(self):
        cfg = AudioConfig(max_recording_duration_s=30, chunk_duration_ms=30)
        assert cfg.max_recording_chunks == 1000

    def test_validate_valid(self):
        cfg = AudioConfig(sample_rate=16000, chunk_duration_ms=30)
        assert cfg.validate() is True

    def test_validate_invalid_sample_rate(self):
        cfg = AudioConfig(sample_rate=11025, chunk_duration_ms=30)
        assert cfg.validate() is False

    def test_validate_invalid_chunk_duration(self):
        cfg = AudioConfig(sample_rate=16000, chunk_duration_ms=25)
        assert cfg.validate() is False

    def test_validate_all_valid_rates_8000(self):
        assert AudioConfig(sample_rate=8000, chunk_duration_ms=10).validate() is True

    def test_validate_all_valid_rates_16000(self):
        assert AudioConfig(sample_rate=16000, chunk_duration_ms=10).validate() is True

    def test_validate_all_valid_rates_32000(self):
        assert AudioConfig(sample_rate=32000, chunk_duration_ms=10).validate() is True

    def test_validate_all_valid_rates_48000(self):
        assert AudioConfig(sample_rate=48000, chunk_duration_ms=10).validate() is True

    def test_validate_all_valid_chunk_durations_10ms(self):
        assert AudioConfig(sample_rate=16000, chunk_duration_ms=10).validate() is True

    def test_validate_all_valid_chunk_durations_20ms(self):
        assert AudioConfig(sample_rate=16000, chunk_duration_ms=20).validate() is True

    def test_validate_all_valid_chunk_durations_30ms(self):
        assert AudioConfig(sample_rate=16000, chunk_duration_ms=30).validate() is True

    def test_custom_device(self):
        cfg = AudioConfig(device=2)
        assert cfg.device == 2

    def test_energy_threshold_stored(self):
        cfg = AudioConfig(energy_threshold=500.0)
        assert cfg.energy_threshold == 500.0


# ===========================================================================
# AudioState tests
# ===========================================================================

class TestAudioState:

    def test_state_constants(self):
        assert AudioState.IDLE == "idle"
        assert AudioState.LISTENING == "listening"
        assert AudioState.SPEECH_DETECTED == "speech_detected"
        assert AudioState.RECORDING == "recording"
        assert AudioState.PROCESSING == "processing"
        assert AudioState.STOPPED == "stopped"


# ===========================================================================
# RealtimeAudioCapture – construction & state
# ===========================================================================

class TestRealtimeAudioCaptureInit:

    def test_default_construction(self):
        cap = _make_capture()
        assert cap.state == AudioState.IDLE
        assert cap.on_audio_ready is None

    def test_invalid_config_raises(self):
        bad_cfg = AudioConfig(sample_rate=11025)  # invalid
        with pytest.raises(ValueError):
            RealtimeAudioCapture(config=bad_cfg)

    def test_callback_stored(self):
        cb = MagicMock()
        cap = _make_capture(callback=cb)
        assert cap.on_audio_ready is cb

    def test_initial_stats(self):
        cap = _make_capture()
        stats = cap.get_stats()
        assert stats['total_recordings'] == 0
        assert stats['total_audio_seconds'] == 0.0
        assert stats['vad_errors'] == 0
        assert stats['stream_errors'] == 0

    def test_pre_buffer_maxlen(self):
        cfg = AudioConfig(pre_buffer_duration_ms=300, chunk_duration_ms=30)
        cap = _make_capture(config=cfg)
        assert cap.pre_buffer.maxlen == cfg.pre_buffer_chunks


# ===========================================================================
# _set_state / _get_state
# ===========================================================================

class TestStateTransitions:

    def test_set_and_get_state(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        assert cap._get_state() == AudioState.LISTENING

    def test_same_state_transition_no_log(self):
        """Transitioning to the same state must not call logger.debug."""
        cap = _make_capture()
        with patch("audio_capture.logger") as mock_log:
            cap._set_state(AudioState.IDLE)   # already IDLE – no-op
            mock_log.debug.assert_not_called()

    def test_different_state_triggers_debug_log(self):
        """Transitioning to a new state must call logger.debug exactly once."""
        cap = _make_capture()
        with patch("audio_capture.logger") as mock_log:
            cap._set_state(AudioState.LISTENING)
            mock_log.debug.assert_called_once()


# ===========================================================================
# _is_speech
# ===========================================================================

class TestIsSpeech:

    def test_vad_false_no_energy(self):
        cap = _make_capture()
        cap.vad.is_speech.return_value = False
        chunk = np.zeros(480, dtype=np.int16).tobytes()
        assert cap._is_speech(chunk) is False

    def test_vad_true(self):
        cap = _make_capture()
        cap.vad.is_speech.return_value = True
        chunk = np.zeros(480, dtype=np.int16).tobytes()
        assert cap._is_speech(chunk) is True

    def test_energy_threshold_triggers_speech(self):
        cfg = AudioConfig(energy_threshold=100.0)
        cap = _make_capture(config=cfg)
        cap.vad.is_speech.return_value = False
        # int16 amplitude 10000 → RMS = 10000.0 which exceeds threshold 100.0
        loud = (np.ones(480, dtype=np.int16) * 10000).tobytes()
        arr = np.frombuffer(loud, dtype=np.int16)
        energy = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
        assert energy > cfg.energy_threshold
        assert cap._is_speech(loud)

    def test_energy_threshold_low_energy_not_speech(self):
        cfg = AudioConfig(energy_threshold=10000.0)
        cap = _make_capture(config=cfg)
        cap.vad.is_speech.return_value = False
        # silent audio → energy == 0.0 < 10000.0
        silent = np.zeros(480, dtype=np.int16).tobytes()
        arr = np.frombuffer(silent, dtype=np.int16)
        energy = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
        assert energy < cfg.energy_threshold
        assert not cap._is_speech(silent)

    def test_vad_exception_increments_counter(self):
        cap = _make_capture()
        cap.vad.is_speech.side_effect = Exception("boom")
        chunk = np.zeros(480, dtype=np.int16).tobytes()
        result = cap._is_speech(chunk)
        assert result is False
        assert cap.stats['vad_errors'] == 1

    def test_vad_or_energy_either_true(self):
        """When VAD is False but energy exceeds threshold → speech."""
        cfg = AudioConfig(energy_threshold=50.0)
        cap = _make_capture(config=cfg)
        cap.vad.is_speech.return_value = False
        # int16 amplitude 200 → RMS 200 > threshold 50
        loud = (np.ones(480, dtype=np.int16) * 200).tobytes()
        arr = np.frombuffer(loud, dtype=np.int16)
        assert float(np.sqrt(np.mean(arr.astype(np.float32)**2))) > cfg.energy_threshold
        assert cap._is_speech(loud)


# ===========================================================================
# _process_audio_chunk
# ===========================================================================

class TestProcessAudioChunk:

    def _silent_chunk(self, cap):
        cap.vad.is_speech.return_value = False
        return np.zeros(cap.config.chunk_size, dtype=np.int16).tobytes()

    def _speech_chunk(self, cap):
        cap.vad.is_speech.return_value = True
        return np.zeros(cap.config.chunk_size, dtype=np.int16).tobytes()

    # --- LISTENING state ---

    def test_listening_silent_stays_listening(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        cap._process_audio_chunk(self._silent_chunk(cap))
        assert cap._get_state() == AudioState.LISTENING

    def test_listening_silent_grows_prebuffer(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        cap._process_audio_chunk(self._silent_chunk(cap))
        assert len(cap.pre_buffer) == 1

    def test_listening_speech_transitions_to_speech_detected(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        cap._process_audio_chunk(self._speech_chunk(cap))
        assert cap._get_state() in (AudioState.SPEECH_DETECTED, AudioState.RECORDING)

    def test_listening_speech_calls_start_recording(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        cap._start_recording = MagicMock(side_effect=cap._start_recording)
        cap._process_audio_chunk(self._speech_chunk(cap))
        cap._start_recording.assert_called_once()

    # --- SPEECH_DETECTED state ---

    def test_speech_detected_transitions_to_recording(self):
        cap = _make_capture()
        cap._set_state(AudioState.SPEECH_DETECTED)
        cap._process_audio_chunk(self._silent_chunk(cap))
        assert cap._get_state() == AudioState.RECORDING

    def test_speech_detected_appends_to_buffer(self):
        cap = _make_capture()
        cap._set_state(AudioState.SPEECH_DETECTED)
        cap._process_audio_chunk(self._silent_chunk(cap))
        assert cap.recording_chunks_count == 1

    # --- RECORDING state ---

    def test_recording_speech_resets_silence_counter(self):
        cap = _make_capture()
        cap._set_state(AudioState.RECORDING)
        cap.silence_counter = 10
        cap._process_audio_chunk(self._speech_chunk(cap))
        assert cap.silence_counter == 0

    def test_recording_silence_increments_counter(self):
        cap = _make_capture()
        cap._set_state(AudioState.RECORDING)
        cap._process_audio_chunk(self._silent_chunk(cap))
        assert cap.silence_counter == 1

    def test_recording_ends_after_silence_threshold(self):
        cap = _make_capture()
        cap._set_state(AudioState.RECORDING)
        cap.silence_counter = cap.config.silence_chunks - 1
        cap._end_recording = MagicMock()
        cap._process_audio_chunk(self._silent_chunk(cap))
        cap._end_recording.assert_called_once()

    def test_recording_ends_after_max_duration(self):
        cap = _make_capture()
        cap._set_state(AudioState.RECORDING)
        cap.recording_chunks_count = cap.config.max_recording_chunks - 1
        cap._end_recording = MagicMock()
        cap._process_audio_chunk(self._speech_chunk(cap))
        cap._end_recording.assert_called_once()

    def test_recording_appends_chunks(self):
        cap = _make_capture()
        cap._set_state(AudioState.RECORDING)
        before = len(cap.recording_buffer)
        cap._process_audio_chunk(self._silent_chunk(cap))
        assert len(cap.recording_buffer) == before + 1


# ===========================================================================
# _start_recording
# ===========================================================================

class TestStartRecording:

    def test_copies_pre_buffer(self):
        cap = _make_capture()
        chunk = b'\x00' * (cap.config.chunk_size * 2)
        cap.pre_buffer.append(chunk)
        cap._start_recording()
        assert chunk in cap.recording_buffer

    def test_recording_chunk_count_matches_prebuffer(self):
        cap = _make_capture()
        for _ in range(3):
            cap.pre_buffer.append(b'\x00' * 10)
        cap._start_recording()
        assert cap.recording_chunks_count == 3

    def test_silence_counter_reset(self):
        cap = _make_capture()
        cap.silence_counter = 99
        cap._start_recording()
        assert cap.silence_counter == 0


# ===========================================================================
# _end_recording
# ===========================================================================

class TestEndRecording:

    def _fill_recording(self, cap, seconds=0.5):
        n_chunks = int(seconds * 1000 / cap.config.chunk_duration_ms)
        chunk = np.zeros(cap.config.chunk_size, dtype=np.int16).tobytes()
        cap.recording_buffer = [chunk] * n_chunks
        cap.recording_chunks_count = n_chunks

    def test_callback_invoked(self):
        cb = MagicMock()
        cap = _make_capture(callback=cb)
        self._fill_recording(cap)
        cap._end_recording()
        cb.assert_called_once()

    def test_callback_receives_float32_array(self):
        cb = MagicMock()
        cap = _make_capture(callback=cb)
        self._fill_recording(cap)
        cap._end_recording()
        audio_arg = cb.call_args[0][0]
        assert audio_arg.dtype == np.float32

    def test_callback_receives_metadata(self):
        cb = MagicMock()
        cap = _make_capture(callback=cb)
        self._fill_recording(cap)
        cap._end_recording()
        meta = cb.call_args[0][1]
        assert 'duration_s' in meta
        assert 'sample_rate' in meta
        assert 'num_samples' in meta
        assert 'timestamp' in meta

    def test_stats_total_recordings_incremented(self):
        cap = _make_capture()
        self._fill_recording(cap)
        cap._end_recording()
        assert cap.stats['total_recordings'] == 1

    def test_stats_total_audio_seconds_updated(self):
        cap = _make_capture()
        self._fill_recording(cap, seconds=1.0)
        cap._end_recording()
        assert cap.stats['total_audio_seconds'] > 0

    def test_buffers_cleared_after_recording(self):
        cap = _make_capture()
        self._fill_recording(cap)
        cap._end_recording()
        assert cap.recording_buffer == []
        assert cap.recording_chunks_count == 0
        assert cap.silence_counter == 0

    def test_state_returns_to_listening(self):
        cap = _make_capture()
        self._fill_recording(cap)
        cap._end_recording()
        assert cap._get_state() == AudioState.LISTENING

    def test_callback_exception_is_caught(self):
        cb = MagicMock(side_effect=RuntimeError("oops"))
        cap = _make_capture(callback=cb)
        self._fill_recording(cap)
        cap._end_recording()   # must not raise
        assert cap._get_state() == AudioState.LISTENING

    def test_no_callback_no_error(self):
        cap = _make_capture(callback=None)
        self._fill_recording(cap)
        cap._end_recording()   # should not raise


# ===========================================================================
# _audio_callback
# ===========================================================================

class TestAudioCallback:

    def _fake_indata(self, cap, value=0.0):
        return np.full((cap.config.chunk_size, 1), value, dtype=np.float32)

    def test_stopped_state_returns_early(self):
        cap = _make_capture()
        cap._set_state(AudioState.STOPPED)
        cap._process_audio_chunk = MagicMock()
        cap._audio_callback(self._fake_indata(cap), cap.config.chunk_size, {}, None)
        cap._process_audio_chunk.assert_not_called()

    def test_stream_error_increments_counter(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        cap._process_audio_chunk = MagicMock()
        cap._audio_callback(self._fake_indata(cap), cap.config.chunk_size, {}, "xrun")
        assert cap.stats['stream_errors'] == 1

    def test_no_status_no_error_increment(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        cap._process_audio_chunk = MagicMock()
        cap._audio_callback(self._fake_indata(cap), cap.config.chunk_size, {}, None)
        assert cap.stats['stream_errors'] == 0

    def test_passes_bytes_to_process_chunk(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        cap._process_audio_chunk = MagicMock()
        cap._audio_callback(self._fake_indata(cap), cap.config.chunk_size, {}, None)
        cap._process_audio_chunk.assert_called_once()
        arg = cap._process_audio_chunk.call_args[0][0]
        assert isinstance(arg, bytes)


# ===========================================================================
# start / stop
# ===========================================================================

class TestStartStop:

    def test_start_sets_listening_state(self):
        cap = _make_capture()
        with patch("sounddevice.InputStream") as MockStream:
            MockStream.return_value.__enter__ = MagicMock(return_value=MockStream.return_value)
            mock_instance = MagicMock()
            MockStream.return_value = mock_instance
            cap.start()
        assert cap._get_state() == AudioState.LISTENING

    def test_start_when_already_running_is_noop(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        with patch("sounddevice.InputStream") as MockStream:
            cap.start()
            MockStream.assert_not_called()

    def test_start_stream_failure_reverts_to_idle(self):
        cap = _make_capture()
        import sounddevice as sd
        original = sd.InputStream
        sd.InputStream = MagicMock(side_effect=OSError("no device"))
        try:
            with pytest.raises(OSError):
                cap.start()
            assert cap._get_state() == AudioState.IDLE
        finally:
            sd.InputStream = original

    def test_stop_sets_idle(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        mock_stream = MagicMock()
        cap.stream = mock_stream
        cap.stop()
        assert cap._get_state() == AudioState.IDLE

    def test_stop_calls_stream_stop_and_close(self):
        cap = _make_capture()
        cap._set_state(AudioState.LISTENING)
        mock_stream = MagicMock()
        cap.stream = mock_stream
        cap.stop()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


# ===========================================================================
# get_stats
# ===========================================================================

class TestGetStats:

    def test_returns_copy(self):
        cap = _make_capture()
        s1 = cap.get_stats()
        s1['total_recordings'] = 999
        assert cap.stats['total_recordings'] == 0

    def test_stats_keys_present(self):
        cap = _make_capture()
        stats = cap.get_stats()
        for key in ('total_recordings', 'total_audio_seconds', 'vad_errors', 'stream_errors'):
            assert key in stats


# ===========================================================================
# list_audio_devices
# ===========================================================================

class TestListAudioDevices:

    def test_returns_only_input_devices(self):
        import sounddevice as sd
        sd.query_devices.return_value = [
            {'name': 'Mic', 'max_input_channels': 2, 'max_output_channels': 0},
            {'name': 'Speakers', 'max_input_channels': 0, 'max_output_channels': 2},
        ]
        result = RealtimeAudioCapture.list_audio_devices()
        assert len(result) == 1
        assert result[0][1]['name'] == 'Mic'

    def test_empty_device_list(self):
        import sounddevice as sd
        sd.query_devices.return_value = []
        result = RealtimeAudioCapture.list_audio_devices()
        assert result == []

    def test_multiple_input_devices(self):
        import sounddevice as sd
        sd.query_devices.return_value = [
            {'name': 'Mic A', 'max_input_channels': 1, 'max_output_channels': 0},
            {'name': 'Mic B', 'max_input_channels': 2, 'max_output_channels': 0},
        ]
        result = RealtimeAudioCapture.list_audio_devices()
        assert len(result) == 2


# ===========================================================================
# Integration-style: full speech detection → callback flow
# ===========================================================================

class TestFullFlow:

    def _make_int16_chunk(self, cap):
        return np.zeros(cap.config.chunk_size, dtype=np.int16).tobytes()

    def test_full_speech_then_silence_triggers_callback(self):
        received = []

        def cb(audio, meta):
            received.append((audio, meta))

        cap = _make_capture(callback=cb)
        cap._set_state(AudioState.LISTENING)

        # Trigger speech
        cap.vad.is_speech.return_value = True
        cap._process_audio_chunk(self._make_int16_chunk(cap))

        # Keep recording for a few chunks
        for _ in range(3):
            cap._process_audio_chunk(self._make_int16_chunk(cap))

        # Now silence to trigger end
        cap.vad.is_speech.return_value = False
        silence_needed = cap.config.silence_chunks
        for _ in range(silence_needed):
            cap._process_audio_chunk(self._make_int16_chunk(cap))

        assert len(received) == 1
        audio, meta = received[0]
        assert isinstance(audio, np.ndarray)
        assert meta['sample_rate'] == cap.config.sample_rate

    def test_multiple_utterances_multiple_callbacks(self):
        received = []
        cap = _make_capture(callback=lambda a, m: received.append(a))

        def do_utterance():
            cap._set_state(AudioState.LISTENING)
            chunk = self._make_int16_chunk(cap)
            # trigger speech → SPEECH_DETECTED
            cap.vad.is_speech.return_value = True
            cap._process_audio_chunk(chunk)
            # one more chunk in RECORDING
            cap._process_audio_chunk(chunk)
            # silence to end recording
            cap.vad.is_speech.return_value = False
            for _ in range(cap.config.silence_chunks):
                cap._process_audio_chunk(chunk)

        do_utterance()
        do_utterance()
        assert len(received) == 2