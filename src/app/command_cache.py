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
        """Return cached command or None. Lookup order: checks normalized key then verbatim fallback."""
        sk = self._clean_key(transcript)
        vk = transcript.lower().strip()
        with self._lock:
            result = self._store.get(sk) or (self._store.get(vk) if vk != sk else None)
        if result:
            logger.info("Cache HIT  '%s' -> '%s'", sk, result)
        else:
            logger.debug("Cache MISS '%s'", sk)
        return result

    def set(self, transcript: str, clarified_command: str) -> None:
        """
        Store clarified_command for transcript and flush to disk.
        transcript should be pre-normalised (pass _root_transcript, not raw input).
        Idempotent : no-op if the mapping already exists.
        """
        sk = self._clean_key(transcript)
        vk = transcript.lower().strip()

        with self._lock:
            existing = self._store.get(sk)
            if existing and existing != clarified_command:
                if vk != sk:
                    key = vk  # Store under verbatim key on normalization collision
                    logger.warning("Collision on '%s' — storing under verbatim key.", sk)
                else:
                    logger.warning("Collision on '%s' — keeping existing mapping.", sk)
                    return
            else:
                key = sk

            if self._store.get(key) == clarified_command:
                return

            self._store[key] = clarified_command
            self._flush()

        logger.info("Cache SET  '%s' -> '%s'", key, clarified_command)

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
            self._store = {k: str(v) for k, v in data.items()}
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