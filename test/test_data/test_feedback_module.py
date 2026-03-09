import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from feedback_store import FeedbackStore  # module under test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """FeedbackStore backed by a real temp file that is cleaned up automatically."""
    return FeedbackStore(log_path=tmp_path / "feedback_log.jsonl")


def _read_records(store: FeedbackStore) -> list[dict]:
    """Read all JSONL records written to the store's log file."""
    return [json.loads(line) for line in store.log_path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:

    def test_explicit_log_path_is_used(self, tmp_path):
        """Constructor should use the path that is explicitly passed in."""
        p = tmp_path / "custom.jsonl"
        store = FeedbackStore(log_path=p)
        assert store.log_path == p

    def test_default_log_path_is_sibling_of_module(self):
        """Without an explicit path, log file should sit in the same directory as the module."""
        import feedback_store as _fm
        store = FeedbackStore()
        assert store.log_path.parent == Path(_fm.__file__).parent

    def test_default_log_path_name(self):
        """Default log file should be named feedback_log.jsonl."""
        store = FeedbackStore()
        assert store.log_path.name == "feedback_log.jsonl"

    def test_log_path_stored_as_path_object(self, tmp_path):
        """log_path should be a Path even when a string is supplied."""
        store = FeedbackStore(log_path=str(tmp_path / "x.jsonl"))
        assert isinstance(store.log_path, Path)

    def test_lock_is_rlock(self):
        """_lock should be a re-entrant lock."""
        store = FeedbackStore()
        assert isinstance(store._lock, type(threading.RLock()))


# ---------------------------------------------------------------------------
# Record structure
# ---------------------------------------------------------------------------

class TestRecordStructure:

    def test_record_has_all_required_keys(self, tmp_store):
        """Every written record must contain the seven expected keys."""
        tmp_store.log_feedback("cmd-1", "rating", "yes")
        record = _read_records(tmp_store)[0]
        assert {"uuid", "command_id", "command_text", "type", "value", "source", "timestamp"} \
               <= record.keys()

    def test_uuid_is_valid(self, tmp_store):
        """uuid field must be a valid UUID-4 string."""
        tmp_store.log_feedback("cmd-1", "rating", "yes")
        record = _read_records(tmp_store)[0]
        parsed = uuid.UUID(record["uuid"])
        assert parsed.version == 4

    def test_two_records_have_different_uuids(self, tmp_store):
        """Each call should generate a unique uuid."""
        tmp_store.log_feedback("cmd-1", "rating", "yes")
        tmp_store.log_feedback("cmd-2", "rating", "no")
        records = _read_records(tmp_store)
        assert records[0]["uuid"] != records[1]["uuid"]

    def test_command_id_is_stored(self, tmp_store):
        """command_id should be written verbatim."""
        tmp_store.log_feedback("my-cmd-99", "rating", "yes")
        assert _read_records(tmp_store)[0]["command_id"] == "my-cmd-99"

    def test_feedback_type_is_stored(self, tmp_store):
        """type field should match the feedback_type argument."""
        tmp_store.log_feedback("c1", "satisfaction", "yes")
        assert _read_records(tmp_store)[0]["type"] == "satisfaction"

    def test_value_is_stored(self, tmp_store):
        """value field should match the value argument."""
        tmp_store.log_feedback("c1", "rating", "up")
        assert _read_records(tmp_store)[0]["value"] == "up"

    def test_command_text_is_stored(self, tmp_store):
        """command_text should be written when supplied."""
        tmp_store.log_feedback("c1", "rating", "yes", command_text="open browser")
        assert _read_records(tmp_store)[0]["command_text"] == "open browser"

    def test_command_text_defaults_to_none(self, tmp_store):
        """command_text should be null when not provided."""
        tmp_store.log_feedback("c1", "rating", "yes")
        assert _read_records(tmp_store)[0]["command_text"] is None

    # CHANGED: datetime.now(timezone.utc).isoformat() produces "+00:00" not "Z"
    def test_timestamp_ends_with_utc_offset(self, tmp_store):
        """timestamp must end with '+00:00' (timezone-aware isoformat)."""
        tmp_store.log_feedback("c1", "rating", "yes")
        assert _read_records(tmp_store)[0]["timestamp"].endswith("+00:00")

    # CHANGED: parse directly without stripping "Z"; assert tzinfo is set
    def test_timestamp_is_valid_iso_format(self, tmp_store):
        """timestamp must be parseable as a timezone-aware ISO-8601 datetime."""
        tmp_store.log_feedback("c1", "rating", "yes")
        ts = _read_records(tmp_store)[0]["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None


# ---------------------------------------------------------------------------
# Source auto-detection
# ---------------------------------------------------------------------------

class TestSourceAutoDetection:

    @pytest.mark.parametrize("value", ["yes", "YES", "Yes"])
    def test_yes_detected_as_verbal(self, tmp_store, value):
        """'yes' in any casing should be classified as verbal."""
        tmp_store.log_feedback("c1", "rating", value)
        assert _read_records(tmp_store)[0]["source"] == "verbal"

    @pytest.mark.parametrize("value", ["no", "NO", "No"])
    def test_no_detected_as_verbal(self, tmp_store, value):
        """'no' in any casing should be classified as verbal."""
        tmp_store.log_feedback("c1", "rating", value)
        assert _read_records(tmp_store)[0]["source"] == "verbal"

    @pytest.mark.parametrize("value", ["up", "down", "thumbs_up", "thumbs_down",
                                        "UP", "DOWN", "THUMBS_UP", "THUMBS_DOWN"])
    def test_ui_values_detected_as_ui(self, tmp_store, value):
        """UI gesture values in any casing should be classified as ui."""
        tmp_store.log_feedback("c1", "rating", value)
        assert _read_records(tmp_store)[0]["source"] == "ui"

    def test_unknown_value_detected_as_unknown(self, tmp_store):
        """An unrecognised value should produce source='unknown'."""
        tmp_store.log_feedback("c1", "rating", "maybe")
        assert _read_records(tmp_store)[0]["source"] == "unknown"

    def test_explicit_source_overrides_auto_detect(self, tmp_store):
        """An explicitly passed source should never be overridden."""
        tmp_store.log_feedback("c1", "rating", "yes", source="ui")
        assert _read_records(tmp_store)[0]["source"] == "ui"

    def test_explicit_source_unknown_string_is_preserved(self, tmp_store):
        """Any explicit source string should be stored as-is."""
        tmp_store.log_feedback("c1", "rating", "up", source="custom_source")
        assert _read_records(tmp_store)[0]["source"] == "custom_source"


# ---------------------------------------------------------------------------
# File writing behaviour
# ---------------------------------------------------------------------------

class TestFileWriting:

    def test_record_appended_as_valid_json(self, tmp_store):
        """Each line in the log file must be valid JSON."""
        tmp_store.log_feedback("c1", "rating", "yes")
        line = tmp_store.log_path.read_text(encoding="utf-8").strip()
        json.loads(line)  # raises if not valid JSON

    def test_multiple_records_each_on_own_line(self, tmp_store):
        """Multiple calls should produce one JSON object per line."""
        for i in range(5):
            tmp_store.log_feedback(f"cmd-{i}", "rating", "yes")
        lines = tmp_store.log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        for line in lines:
            json.loads(line)

    def test_appends_not_overwrites(self, tmp_store):
        """Subsequent calls must append — not overwrite — existing records."""
        tmp_store.log_feedback("c1", "rating", "yes")
        tmp_store.log_feedback("c2", "rating", "no")
        records = _read_records(tmp_store)
        assert len(records) == 2
        assert records[0]["command_id"] == "c1"
        assert records[1]["command_id"] == "c2"

    def test_log_path_created_if_not_exists(self, tmp_path):
        """log_feedback should create the log file if it does not yet exist."""
        store = FeedbackStore(log_path=tmp_path / "new_dir" / "log.jsonl")
        store.log_path.parent.mkdir(parents=True, exist_ok=True)
        store.log_feedback("c1", "rating", "yes")
        assert store.log_path.exists()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    # patch("pathlib.Path.open") targets the method on the class itself,
    # which works on all platforms including Windows where instance-level
    # attribute assignment on Path objects is read-only.

    def test_io_error_does_not_raise(self, tmp_store):
        """An IOError during file write must be swallowed, not propagated."""
        with patch("pathlib.Path.open", side_effect=IOError("disk full")):
            tmp_store.log_feedback("c1", "rating", "yes")  # must not raise

    def test_io_error_is_logged(self, tmp_store):
        """A write failure should be reported via the module logger."""
        with patch("pathlib.Path.open", side_effect=IOError("disk full")), \
             patch("logging.Logger.error") as mock_log:
            tmp_store.log_feedback("c1", "rating", "yes")
        mock_log.assert_called_once()
        assert "disk full" in mock_log.call_args[0][0]

    def test_second_call_succeeds_after_previous_error(self, tmp_store):
        """A failed write should not prevent subsequent successful writes."""
        with patch("pathlib.Path.open", side_effect=IOError("disk full")):
            tmp_store.log_feedback("c1", "rating", "yes")
        # patch is lifted — second call hits the real file and should succeed
        tmp_store.log_feedback("c2", "rating", "no")
        assert len(_read_records(tmp_store)) == 1
        assert _read_records(tmp_store)[0]["command_id"] == "c2"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_writes_all_recorded(self, tmp_store):
        """All records written from concurrent threads must appear in the log."""
        n = 50
        errors = []

        def write(i):
            try:
                tmp_store.log_feedback(f"cmd-{i}", "rating", "yes")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised: {errors}"
        records = _read_records(tmp_store)
        assert len(records) == n

    def test_concurrent_writes_produce_unique_uuids(self, tmp_store):
        """Concurrent writes must each produce a distinct uuid."""
        n = 50

        def write(i):
            tmp_store.log_feedback(f"cmd-{i}", "rating", "yes")

        threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        uuids = [r["uuid"] for r in _read_records(tmp_store)]
        assert len(uuids) == len(set(uuids))