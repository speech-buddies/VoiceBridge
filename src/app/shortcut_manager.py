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
            if self._recording:
                return False
            self._recording = True
            self._buffer = []
            return True

    def add_recorded_command(self, clarified_command: str) -> None:
        with self._lock:
            if not self._recording:
                return
            self._buffer.append(clarified_command)

    def stop_recording(self) -> dict:
        with self._lock:
            if not self._recording:
                raise ValueError("No active shortcut recording")

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