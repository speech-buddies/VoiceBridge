"""
CommandCache — src/app/command_cache.py

Persistent JSON-backed cache mapping a normalized user transcript
to a clarified browser command.

Normalization
    - Removes non-semantic filler words (e.g., "hey", "please", "uh")
    - Preserves verbs and meaningful prepositions (e.g., "to", "on", "for")

        "Hey can you go to Gmail"  ->  "go to gmail"
        "search for cats"          ->  "search for cats"

Collision handling
    If two transcripts normalize to the same key, the newer mapping is
    stored under its verbatim lowercase form. If normalization does not
    change the key, the existing mapping is retained.

Durability
    - Atomic writes (temp file + os.replace)
    - threading.Lock protects concurrent access (e.g., VAD thread / FastAPI)
"""

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("CommandCache")

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "command_cache.json"

# Words removed during key normalization.
# Verbs and meaningful prepositions are intentionally preserved.
_IGNORED_TOKENS: frozenset = frozenset({
    "hey", "hi", "hello", "please", "can", "you", "uh", "um", "like", "a", "an", "the",
})


class CommandCache:
    """Persistent transcript -> clarified-command cache with normalisation and collision handling."""

    def __init__(self, path=None):
        self._path = Path(path) if path else _DEFAULT_PATH
        self._lock = threading.Lock()
        self._store: dict = {}
        self._load()

    def get(self, transcript: str) -> Optional[str]:
        """Return cached clarified command or None."""
        sk = self._clean_key(transcript)
        vk = transcript.lower().strip()

        with self._lock:
            entry = self._store.get(sk) or (self._store.get(vk) if vk != sk else None)

        if not entry:
            logger.debug("Cache MISS '%s'", sk)
            return None

        if isinstance(entry, str):
            result = entry
        else:
            result = entry.get("clarified_command")

        logger.info("Cache HIT  '%s' -> '%s'", sk, result)
        return result
    
    def get_entry(self, transcript: str) -> Optional[dict]:
        """Return full cached entry including shortcut flags."""
        sk = self._clean_key(transcript)
        vk = transcript.lower().strip()

        with self._lock:
            entry = self._store.get(sk) or (self._store.get(vk) if vk != sk else None)

        if not entry:
            return None

        if isinstance(entry, str):
            return {
                "clarified_command": entry,
                "shortcut_start": False,
                "shortcut_end": False,
            }

        return entry

    def set(
        self,
        transcript: str,
        clarified_command: str,
        shortcut_start: bool = False,
        shortcut_end: bool = False,
    ) -> None:
        """
        Store clarified_command for transcript and flush to disk.
        """
        sk = self._clean_key(transcript)
        vk = transcript.lower().strip()

        new_entry = {
            "clarified_command": clarified_command,
            "shortcut_start": shortcut_start,
            "shortcut_end": shortcut_end,
        }

        with self._lock:
            existing = self._store.get(sk)

            existing_command = None
            if isinstance(existing, str):
                existing_command = existing
            elif isinstance(existing, dict):
                existing_command = existing.get("clarified_command")

            if existing_command and existing_command != clarified_command:
                if vk != sk:
                    key = vk
                    logger.warning("Collision on '%s' — storing under verbatim key.", sk)
                else:
                    logger.warning("Collision on '%s' — keeping existing mapping.", sk)
                    return
            else:
                key = sk

            current = self._store.get(key)
            if current == new_entry:
                return

            self._store[key] = new_entry
            self._flush()

        logger.info("Cache SET '%s' -> '%s' [start=%s, end=%s]",
                    key, clarified_command, shortcut_start, shortcut_end)

    def invalidate(self, transcript: str) -> bool:
        """Remove all cached entries for transcript.

        Returns True if at least one entry was removed.
        """
        sk = self._clean_key(transcript)
        vk = transcript.lower().strip()
        with self._lock:
            removed = bool(self._store.pop(sk, None))
            if vk != sk:
                removed |= bool(self._store.pop(vk, None))
            if removed:
                self._flush()
        if removed:
            logger.info("Cache INVALIDATED '%s'", transcript)
        return removed

    def clear(self) -> int:
        """Remove all cached entries and delete the cache file.

        Returns the number of entries removed.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            try:
                self._path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete cache file: %s", exc)
        logger.info("Cache CLEARED (%d entries removed)", count)
        return count

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def path(self) -> Path:
        return self._path

    # ── Normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_key(text: str) -> str:
        """
            Normalize a transcript for use as a cache key.

            - Lowercase
            - Remove trailing sentence punctuation
            - Remove ignored tokens

            "Hey can you please open Gmail"  ->  "open gmail"
            "hi please go to youtube"        ->  "go to youtube"
            "open email."                    ->  "open email"
        """
        # Remove trailing punctuation per token to avoid STT-induced key mismatches.
        _PUNCT = str.maketrans("", "", ".,!?;:")
        tokens = text.lower().split()
        tokens = [t.translate(_PUNCT) for t in tokens]
        kept = [t for t in tokens if t and t not in _IGNORED_TOKENS]
        return " ".join(kept) if kept else text.lower().strip()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load from disk on startup. Silent on missing file; warns and starts fresh on corruption."""
        if not self._path.exists():
            logger.info("No cache file found at %s; starting empty", self._path)
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            normalized_store = {}
            for k, v in data.items():
                if isinstance(v, str):
                    normalized_store[k] = v
                elif isinstance(v, dict):
                    normalized_store[k] = {
                        "clarified_command": str(v.get("clarified_command", "")),
                        "shortcut_start": bool(v.get("shortcut_start", False)),
                        "shortcut_end": bool(v.get("shortcut_end", False)),
                    }

            self._store = normalized_store
            logger.info("Loaded %d entries from %s", len(self._store), self._path)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning("Cache unreadable (%s); starting empty", exc)
            self._store = {}

    def _flush(self) -> None:
        """Atomically write store to disk via temp file + os.replace. Caller must hold _lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".cmd_cache_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._store, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self._path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.error("Failed to write cache: %s", exc)