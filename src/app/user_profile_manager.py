"""
UserProfileManager — src/app/user_profile_manager.py

Manages a single-user preference profile and training state for VoiceBridge.

Storage layout (both files written to the project-level data/ directory):
    data/preferences.json   — JSON object: { profile_id: PreferenceData, ... }
    data/consent_log.jsonl  — Append-only JSONL audit log of every consent change

Thread safety
    All public methods are protected by a single RLock so they are safe to
    call from both the FastAPI request thread and the VAD audio thread.

Atomic writes
    preferences.json is written via temp-file + os.replace, matching the
    pattern used by CommandCache._flush().

Author: Luna Aljammal
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("UserProfileManager")

PreferenceData = Dict[str, Any]

DEFAULT_PROFILE_ID = "default"

_DATA_DIR           = Path(__file__).resolve().parent.parent.parent / "data"
_PREFERENCES_PATH   = _DATA_DIR / "preferences.json"
_CONSENT_LOG_PATH   = _DATA_DIR / "consent_log.jsonl"

# User-settable fields (not training state)
_SETTABLE_FIELDS: frozenset = frozenset({
    "guardrails_enabled",
    "custom_training_enabled",
    "custom_shortcuts_enabled",
})

# Full schema with defaults. New fields here auto-merge into existing profiles on _load().
_PREFERENCE_DEFAULTS: PreferenceData = {
    # User-settable 
    "guardrails_enabled":        True,
    "custom_training_enabled":   False,
    "custom_shortcuts_enabled":  False,
    # Server-managed training state
    "training_in_progress":      False,
    "training_completed":        False,
    "training_progress_pct":     0,       # 0–100
    "accumulated_audio_seconds": 0,       # canonical source of truth
}


class ProfileCreationError(Exception):
    """Raised when a new profile cannot be created or persisted."""


class ProfileNotFoundError(Exception):
    """Raised when an operation references a profile_id that does not exist."""

class UserProfileManager:
    """
    Manages user preference profiles and voice-model training state.

    Public interface
    ────────────────
    create_profile(init_data)                       -> str
    load_preferences(profile_id)                    -> PreferenceData
    update_preferences(updates, profile_id)         -> PreferenceData
    set_training_state(**kwargs, profile_id)        -> PreferenceData
    save_consent(profile_id, consent_flag)          -> bool
    """

    def __init__(
        self,
        preferences_path: Optional[Path] = None,
        consent_log_path: Optional[Path] = None,
    ):
        self._preferences_path = Path(preferences_path) if preferences_path else _PREFERENCES_PATH
        self._consent_log_path = Path(consent_log_path) if consent_log_path else _CONSENT_LOG_PATH
        self._lock = threading.RLock()
        self._store: Dict[str, PreferenceData] = {}
        self._load()
        self._ensure_default_profile()

    def create_profile(self, init_data: Dict[str, Any]) -> str:
        """
        Create a new profile pre-populated with init_data (settable fields only).
        Returns the new profile_id. Raises ProfileCreationError if persistence fails.
        """
        profile_id = f"profile_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        prefs = dict(_PREFERENCE_DEFAULTS)
        for field in _SETTABLE_FIELDS:
            if field in init_data:
                prefs[field] = init_data[field]

        with self._lock:
            if profile_id in self._store:
                raise ProfileCreationError(f"Profile ID collision: {profile_id}.")
            self._store[profile_id] = prefs
            try:
                self._flush()
            except OSError as exc:
                del self._store[profile_id]
                raise ProfileCreationError(f"Failed to persist new profile: {exc}") from exc

        logger.info("Profile created: %s", profile_id)
        return profile_id

    def load_preferences(self, profile_id: str = DEFAULT_PROFILE_ID) -> PreferenceData:
        """Return a copy of preferences for profile_id. Raises ProfileNotFoundError if missing."""
        with self._lock:
            if profile_id not in self._store:
                raise ProfileNotFoundError(f"Profile not found: {profile_id}")
            return dict(self._store[profile_id])

    def update_preferences(
        self,
        updates: Dict[str, Any],
        profile_id: str = DEFAULT_PROFILE_ID,
    ) -> PreferenceData:
        """
        Apply updates to user-settable fields and persist.
        Non-settable keys are silently dropped. Raises ProfileNotFoundError if missing.
        """
        with self._lock:
            if profile_id not in self._store:
                raise ProfileNotFoundError(f"Profile not found: {profile_id}")
            filtered = {k: v for k, v in updates.items() if k in _SETTABLE_FIELDS}
            if not filtered:
                return dict(self._store[profile_id])
            self._store[profile_id].update(filtered)
            self._flush()
            return dict(self._store[profile_id])

    def set_training_state(
        self,
        *,
        training_in_progress: Optional[bool] = None,
        training_completed: Optional[bool] = None,
        training_progress_pct: Optional[int] = None,
        accumulated_audio_seconds: Optional[int] = None,
        profile_id: str = DEFAULT_PROFILE_ID,
    ) -> PreferenceData:
        """
        Update server-managed training-state fields (keyword-only, partial updates safe).
        Raises ProfileNotFoundError if missing.
        """
        with self._lock:
            if profile_id not in self._store:
                raise ProfileNotFoundError(f"Profile not found: {profile_id}")
            prefs = self._store[profile_id]
            if training_in_progress is not None:
                prefs["training_in_progress"] = bool(training_in_progress)
            if training_completed is not None:
                prefs["training_completed"] = bool(training_completed)
            if training_progress_pct is not None:
                prefs["training_progress_pct"] = max(0, min(100, int(training_progress_pct)))
            if accumulated_audio_seconds is not None:
                prefs["accumulated_audio_seconds"] = max(0, int(accumulated_audio_seconds))
            self._flush()
            return dict(prefs)

    def save_consent(
        self,
        profile_id: str = DEFAULT_PROFILE_ID,
        consent_flag: bool = True,
    ) -> bool:
        """
        Log consent change and update preferences. Resets settable fields if revoked.
        """
        with self._lock:
            if profile_id not in self._store:
                raise ProfileNotFoundError(f"Profile not found: {profile_id}")
            self._append_consent_log(profile_id, consent_flag)
            if not consent_flag:
                for field in _SETTABLE_FIELDS:
                    self._store[profile_id][field] = _PREFERENCE_DEFAULTS[field]
                self._flush()
                logger.info("Consent revoked for %s; settable prefs reset.", profile_id)
            else:
                logger.info("Consent granted for %s.", profile_id)
        return True

    def _ensure_default_profile(self) -> None:
        """Bootstrap the default profile on first run."""
        with self._lock:
            if DEFAULT_PROFILE_ID not in self._store:
                self._store[DEFAULT_PROFILE_ID] = dict(_PREFERENCE_DEFAULTS)
                self._flush()
                logger.info("Default profile bootstrapped.")

    def _load(self) -> None:
        """Load preferences.json; merges new schema fields into existing profiles."""
        if not self._preferences_path.exists():
            logger.info("No preferences file at %s — starting empty.", self._preferences_path)
            return
        try:
            raw = json.loads(self._preferences_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Expected a JSON object at top level.")
            for pid, data in raw.items():
                if isinstance(data, dict):
                    merged = dict(_PREFERENCE_DEFAULTS)
                    merged.update(data)
                    self._store[pid] = merged
            logger.info("Loaded %d profile(s) from %s.", len(self._store), self._preferences_path)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning("preferences.json unreadable (%s) — starting empty.", exc)
            self._store = {}

    def _flush(self) -> None:
        """Atomic write via temp-file + os.replace. Caller must hold self._lock."""
        try:
            self._preferences_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=self._preferences_path.parent, prefix=".prefs_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._store, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self._preferences_path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.error("Failed to write preferences.json: %s", exc)
            raise

    def _append_consent_log(self, profile_id: str, consent_flag: bool) -> None:
        """Append one JSON line to consent_log.jsonl. Best-effort — never raises."""
        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "profile_id": profile_id,
            "consent":    consent_flag,
        }
        try:
            self._consent_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._consent_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Could not write consent_log.jsonl: %s", exc)