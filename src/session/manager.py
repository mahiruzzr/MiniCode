"""Session management for conversation history."""

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config.loader import minicode_home

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    unsafe = '<>:"/\\|?*'
    for char in unsafe:
        name = name.replace(char, "_")
    return name.strip()


def _new_session_id() -> str:
    """Generate a short unique identifier for a new session."""
    return uuid.uuid4().hex[:8]


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.
    """

    session_id: str = field(default_factory=_new_session_id)
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        self.messages.append(msg)
        self.updated_at = datetime.now()

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start:i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self) -> None:
        self.sessions_dir = _ensure_dir(minicode_home() / "sessions")
        self._lock = threading.RLock()
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        safe_id = _safe_filename(session_id)
        return self.sessions_dir / f"{safe_id}.jsonl"

    def create_session(
        self,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create and persist a new empty session.

        Args:
            session_id: Explicit session identifier. Generated when
                omitted. Regenerated on collision, never overwriting
                an existing session.
            metadata: Initial session metadata.

        Returns:
            The persisted new session.
        """
        with self._lock:
            candidate = session_id or _new_session_id()
            attempts = 0

            while self._get_session_path(candidate).exists():
                candidate = _new_session_id()
                attempts += 1
                if attempts > 8:
                    raise RuntimeError("Could not allocate a unique session_id")
            session = Session(
                session_id=candidate,
                metadata=dict(metadata or {}),
            )
            self.save(session)
            return session

    def get_session(self, session_id: str) -> Session | None:
        """Load a session from disk, using the cache when available.

        Args:
            session_id: Session identifier.

        Returns:
            The session, or None when not found. Never creates one.
        """
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]
            session = self._load(session_id)
            if session is not None:
                self._cache[session_id] = session
            return session

    def _load(self, session_id: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(session_id)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            updated_at = None

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = (
                            datetime.fromisoformat(data["created_at"])
                            if data.get("created_at")
                            else None
                        )
                        updated_at = (
                            datetime.fromisoformat(data["updated_at"])
                            if data.get("updated_at")
                            else None
                        )
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            return Session(
                session_id=session_id,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.session_id)

        with open(path, "w", encoding="utf-8") as f:
            # Write metadata first
            metadata_line = {
                "_type": "metadata",
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")

            # Write messages
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.session_id] = session

    def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if deleted, False if not found.
        """
        # Remove from cache
        self._cache.pop(session_id, None)

        # Remove file
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            session_id = data.get("session_id", path.stem)
                            sessions.append(
                                {
                                    "session_id": session_id,
                                    "created_at": data.get("created_at"),
                                    "updated_at": data.get("updated_at"),
                                    "path": str(path),
                                }
                            )
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
