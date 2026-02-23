import threading
import uuid
import json
from datetime import datetime
from pathlib import Path

class FeedbackStore:
    def __init__(self, log_path=None):
        self.log_path = Path(log_path) if log_path else Path(__file__).parent / 'feedback_log.jsonl'
        self._lock = threading.RLock()

    def log_feedback(
        self,
        command_id,
        feedback_type,
        value,
        command_text=None,
        source=None  # verbal | ui (auto-detect if None)
    ):
        # Auto-detect source if not explicitly provided
        if source is None:
            if str(value).lower() in ("yes", "no"):
                source = "verbal"
            elif str(value).lower() in ("up", "down", "thumbs_up", "thumbs_down"):
                source = "ui"
            else:
                source = "unknown"

        record = {
            "uuid": str(uuid.uuid4()),
            "command_id": command_id,
            "command_text": command_text,
            "type": feedback_type,
            "value": value,
            "source": source,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            with self._lock:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except Exception as e:
            # Logging must never crash the server
            import logging
            logging.getLogger(__name__).error(
                f"Failed to write feedback log: {e}"
            )