"""
M10: Session Manager (SessionManager)

MG (Module Guide) - M10:
  Secrets:
    - Session lifecycle rules
    - Storage schema
    - TTL and expiry handling
  Services:
    - Start and stop sessions
    - Get session state
    - Attach commands
    - Set state
  Implemented By:
    - In-memory cache + persistent store

MIS (Module Interface Specification) - SessionManager:
  Exported Access Programs:
    SessionManager(store: Store, ttl_s: int) -> self
    start(user_id: UUID) -> SessionId (StoreError)
    stop(session_id: SessionId) -> bool (StoreError)
    get(session_id: SessionId) -> SessionState (NotFoundError)
    attach_command(session_id: SessionId, cmd_id: UUID) -> bool (NotFoundError)
    set_state(session_id: SessionId, state: SessionState) -> bool (StoreError)

Notes:
  - Store ops are assumed atomic per session key (MIS assumption).
  - TTL is idle expiry threshold. Expiry policy is encapsulated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4
import os


# -----------------------
# Types + Exceptions
# -----------------------

SessionId = UUID


class SessionPhase(str, Enum):
    """
    Session states hinted by MG control layer description:
    capture -> transcribe -> confirm -> execute (and terminal states)
    """
    CAPTURE = "capture"
    TRANSCRIBE = "transcribe"
    CONFIRM = "confirm"
    EXECUTE = "execute"
    IDLE = "idle"
    CLOSED = "closed"


@dataclass(frozen=True)
class SessionState:
    """
    Minimal session record that can be cached in-memory and persisted in a store.
    The storage schema is intentionally hidden (MG secret); this is a clean interface object.
    """
    session_id: SessionId
    user_id: UUID
    phase: SessionPhase = SessionPhase.IDLE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attached_commands: List[UUID] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    closed: bool = False


class StoreError(Exception):
    """Raised when the persistent store cannot be read/written (MIS: StoreError)."""


class NotFoundError(Exception):
    """Raised when a session id is unknown (MIS: NotFoundError)."""


# -----------------------
# Store Interface (M12/M17 dependency in MIS wording)
# -----------------------

@runtime_checkable
class Store(Protocol):
    """
    Abstract persistence for sessions.
    Your concrete implementation can be backed by M12 (Storage Management Layer),
    a DB, Redis, or a file store later.

    Atomicity per session key is assumed (MIS).
    """

    def put_session(self, session_id: SessionId, state: SessionState) -> None:
        ...

    def get_session(self, session_id: SessionId) -> Optional[SessionState]:
        ...

    def delete_session(self, session_id: SessionId) -> None:
        ...

    def list_sessions(self) -> List[SessionState]:
        """
        Optional: used to warm cache (MIS mentions 'warm cache from store if available').
        If your store cannot list, return [] or raise StoreError.
        """
        ...


class InMemoryStore(Store):
    """
    Simple placeholder store for development/testing.
    Replace with a real persistence layer wired into M12 later.
    """

    def __init__(self) -> None:
        self._data: Dict[SessionId, SessionState] = {}

    def put_session(self, session_id: SessionId, state: SessionState) -> None:
        self._data[session_id] = state

    def get_session(self, session_id: SessionId) -> Optional[SessionState]:
        return self._data.get(session_id)

    def delete_session(self, session_id: SessionId) -> None:
        self._data.pop(session_id, None)

    def list_sessions(self) -> List[SessionState]:
        return list(self._data.values())


# -----------------------
# Session Manager (M10)
# -----------------------

class SessionManager:
    """
    M10 implementation skeleton: in-memory cache + persistent store.
    """

    def __init__(self, store: Store, ttl_s: Optional[int] = None) -> None:
        """
        MIS: SessionManager(store: Store, ttl_s: int) -> self
        - transition: set fields; warm cache from store if available
        - output: initialized instance
        - exception: -
        """
        self._store = store
        self._ttl_s = int(
            ttl_s if ttl_s is not None else os.getenv("SESSION_TTL_S", "300")
        )
        self._index: Dict[SessionId, SessionState] = {}

        # Warm cache from store (best-effort).
        try:
            for st in self._store.list_sessions():
                self._index[st.session_id] = st
        except Exception:
            # Store listing is optional. Avoid failing construction in skeleton.
            pass

    # ---- Exported access programs (MIS) ----

    def start(self, user_id: UUID) -> SessionId:
        """
        MIS: start(user_id) -> SessionId (StoreError)
        - transition: create session; write to store; cache in index
        """
        session_id = uuid4()
        now = self._now()
        state = SessionState(
            session_id=session_id,
            user_id=user_id,
            phase=SessionPhase.CAPTURE,
            created_at=now,
            last_active_at=now,
        )

        try:
            self._store.put_session(session_id, state)
        except Exception as e:
            raise StoreError(f"Failed to persist session {session_id}") from e

        self._index[session_id] = state
        return session_id

    def stop(self, session_id: SessionId) -> bool:
        """
        MIS: stop(session_id) -> bool (StoreError)
        - transition: mark closed; evict from index; update store
        """
        state = self._get_or_raise(session_id)

        closed_state = self._with_updates(
            state,
            phase=SessionPhase.CLOSED,
            closed=True,
            last_active_at=self._now(),
        )

        try:
            # Persist closed state (or you can delete; policy is hidden inside this module).
            self._store.put_session(session_id, closed_state)
        except Exception as e:
            raise StoreError(f"Failed to stop session {session_id}") from e

        self._index.pop(session_id, None)
        return True

    def get(self, session_id: SessionId) -> SessionState:
        """
        MIS: get(session_id) -> SessionState (NotFoundError)
        - output: current SessionState
        - exception: NotFoundError if unknown
        """
        state = self._get_or_raise(session_id)

        # Expiry is an M10 secret (MG). Policy: if expired, stop/evict and raise NotFound.
        if self._expired(state):
            try:
                self._store.delete_session(session_id)
            except Exception:
                # Best-effort cleanup
                pass
            self._index.pop(session_id, None)
            raise NotFoundError(f"Session expired: {session_id}")

        # Touch last_active_at on read to reflect "idle TTL" semantics
        touched = self._with_updates(state, last_active_at=self._now())
        self._index[session_id] = touched
        try:
            self._store.put_session(session_id, touched)
        except Exception:
            # Don’t fail read if store write fails; but you can tighten later.
            pass

        return touched

    def attach_command(self, session_id: SessionId, cmd_id: UUID) -> bool:
        """
        MIS: attach_command(session_id, cmd_id) -> bool (NotFoundError)
        - transition: append command reference to session state
        """
        state = self._get_or_raise(session_id)
        if self._expired(state) or state.closed:
            raise NotFoundError(f"Session not active: {session_id}")

        updated_cmds = list(state.attached_commands)
        updated_cmds.append(cmd_id)

        new_state = self._with_updates(
            state,
            attached_commands=updated_cmds,
            last_active_at=self._now(),
        )

        self._index[session_id] = new_state
        try:
            self._store.put_session(session_id, new_state)
        except Exception as e:
            raise StoreError(f"Failed to attach command to session {session_id}") from e

        return True

    def set_state(self, session_id: SessionId, state: SessionState) -> bool:
        """
        MIS: set_state(session_id, state) -> bool (StoreError)
        - transition: update state in cache and store
        """
        # Ensure session exists (or allow upsert—policy choice; keep strict to MIS).
        _ = self._get_or_raise(session_id)

        # Normalize: ensure session_id matches key and update last_active_at.
        normalized = self._with_updates(
            state,
            last_active_at=self._now(),
        )
        if normalized.session_id != session_id:
            # Keep deterministic: session_id is the key.
            normalized = self._with_updates(normalized, session_id=session_id)

        self._index[session_id] = normalized
        try:
            self._store.put_session(session_id, normalized)
        except Exception as e:
            raise StoreError(f"Failed to set state for session {session_id}") from e

        return True

    # ---- Local functions (MIS) ----

    @staticmethod
    def _now() -> datetime:
        """MIS local: now(): -> timestamp"""
        return datetime.now(timezone.utc)

    def _expired(self, state: SessionState) -> bool:
        """MIS local: expired(SessionState) -> bool"""
        if state.closed:
            return True
        age_s = (self._now() - state.last_active_at).total_seconds()
        return age_s > self._ttl_s

    # ---- Helpers ----

    def _get_or_raise(self, session_id: SessionId) -> SessionState:
        # 1) Try cache
        cached = self._index.get(session_id)
        if cached is not None:
            return cached

        # 2) Try store
        try:
            st = self._store.get_session(session_id)
        except Exception as e:
            raise StoreError(f"Failed to read session {session_id}") from e

        if st is None:
            raise NotFoundError(f"Unknown session: {session_id}")

        self._index[session_id] = st
        return st

    @staticmethod
    def _with_updates(state: SessionState, **updates: Any) -> SessionState:
        """
        Immutable-update helper for dataclass SessionState.
        """
        data = {
            "session_id": state.session_id,
            "user_id": state.user_id,
            "phase": state.phase,
            "created_at": state.created_at,
            "last_active_at": state.last_active_at,
            "attached_commands": state.attached_commands,
            "metadata": state.metadata,
            "closed": state.closed,
        }
        data.update(updates)
        return SessionState(**data)
