"""In-memory session store with a sliding TTL. No database in v1."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from wwf.api.schemas import BOARD_SIZE, BoardGrid, RecognitionResultOut, SolveResponse

DEFAULT_TTL_S = 30 * 60


def empty_grid() -> BoardGrid:
    return [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]


@dataclass
class Session:
    id: str
    solver_id: str
    params: dict[str, Any]
    dictionary_id: str
    board: BoardGrid = field(default_factory=empty_grid)
    rack: str = ""
    image: bytes | None = None
    image_type: str = "application/octet-stream"
    recognition: RecognitionResultOut | None = None
    corrected: bool = False
    last_result: SolveResponse | None = None
    fixture_name: str | None = None  # set when the picture is a fixture being collected or labelled
    fixture_status: str = "none"  # the label's status when the fixture was opened
    last_access: float = field(default_factory=time.monotonic)


class SessionNotFound(KeyError):
    pass


class SessionStore:
    def __init__(
        self, ttl_s: float = DEFAULT_TTL_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl_s = ttl_s
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, solver_id: str, params: dict[str, Any], dictionary_id: str) -> Session:
        s = Session(
            id=uuid.uuid4().hex,
            solver_id=solver_id,
            params=dict(params),
            dictionary_id=dictionary_id,
            last_access=self._clock(),
        )
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, session_id: str) -> Session:
        """The live session, refreshing its TTL. Raises SessionNotFound if unknown or expired."""
        now = self._clock()
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None or now - s.last_access > self.ttl_s:
                self._sessions.pop(session_id, None)
                raise SessionNotFound(session_id)
            s.last_access = now
            return s

    def expires_in(self, s: Session) -> int:
        return max(0, int(self.ttl_s - (self._clock() - s.last_access)))

    def sweep(self) -> int:
        """Drop expired sessions; returns how many were removed."""
        now = self._clock()
        with self._lock:
            dead = [k for k, s in self._sessions.items() if now - s.last_access > self.ttl_s]
            for k in dead:
                del self._sessions[k]
        return len(dead)

    def __len__(self) -> int:
        return len(self._sessions)
