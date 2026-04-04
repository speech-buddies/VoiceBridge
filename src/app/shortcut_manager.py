"""
Thread-safe manager for recording, storing, and managing reusable command shortcuts.

Shortcuts are sequences of commands captured during a recording session and
persisted to a JSON file. Provides functionality to start/stop recording,
append commands, list existing shortcuts, and delete them.

Author: Mazen Youssef
"""
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger("ShortcutManager")


class ShortcutManager:
    def __init__(self, path: Optional[str] = None):
        self._path = Path(path) if path else Path(__file__).resolve().parent.parent.parent / "shortcuts.json"
        self._lock = Lock()
        self._shortcuts = {}
        self._recording = False
        self._buffer = []
        self._load()

    def start_recording(self) -> bool:
        with self._lock:
            logger.info("Shortcut start requested. Current recording=%s", self._recording)
            if self._recording:
                return False
            self._recording = True
            self._buffer = []
            logger.info("Shortcut recording started.")
            return True

    def add_recorded_command(self, clarified_command: str) -> None:
        with self._lock:
            if not self._recording:
                logger.info("Ignored shortcut command append because recording is off: %s", clarified_command)
                return
            self._buffer.append(clarified_command)
            logger.info("Shortcut buffer append: %s", clarified_command)

    def stop_recording(self) -> dict:
        with self._lock:
            logger.info("Shortcut stop requested. Current recording=%s buffer_len=%d", self._recording, len(self._buffer))
            if not self._recording:
                raise ValueError("There is no active shortcut recording.")

            if not self._buffer:
                self._recording = False
                self._buffer = []
                raise ValueError("No commands were recorded for this shortcut.")

            shortcut_id = str(self._next_id())
            shortcut = {
                "id": shortcut_id,
                "name": f"shortcut_{shortcut_id}",
                "commands": list(self._buffer),
            }

            self._shortcuts[shortcut_id] = shortcut
            self._recording = False
            self._buffer = []
            self._flush()
            logger.info("Shortcut saved successfully: %s", shortcut)
            return shortcut

    def cancel_recording(self) -> None:
        with self._lock:
            self._recording = False
            self._buffer = []

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def list_shortcuts(self) -> dict:
        with self._lock:
            return dict(self._shortcuts)

    def delete_shortcut(self, shortcut_id: str) -> bool:
        with self._lock:
            removed = self._shortcuts.pop(shortcut_id, None) is not None
            if removed:
                self._flush()
            return removed

    def _next_id(self) -> int:
        if not self._shortcuts:
            return 1
        return max(int(k) for k in self._shortcuts.keys()) + 1

    def _load(self) -> None:
        if not self._path.exists():
            self._shortcuts = {}
            return
        try:
            self._shortcuts = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._shortcuts = {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._shortcuts, indent=2), encoding="utf-8")