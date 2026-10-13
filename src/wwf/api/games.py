"""Games: the persistent, multi-turn side of the app.

A Game is a Session that also remembers its history. Each game lives in its own JSON file
(plus its latest picture next to it) so it survives restarts, and a person can play one for days:

    <games dir>/<id>.json    board, rack, turn log, options, last reading and last solve
    <games dir>/<id>.jpg     the latest picture, if any

Board rows are 15 strings: "." empty, uppercase a tile, lowercase a blank. Games started in
fixture-collection mode just to save a picture are "ephemeral": never written, never listed, and
dropped after the session timeout.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from wwf.api.fixtures import grid_to_rows, rows_to_grid
from wwf.api.schemas import BoardGrid, RecognitionResultOut, SolveResponse
from wwf.api.session import Session, SessionNotFound, empty_grid

log = logging.getLogger("wwf.api")

SCHEMA_VERSION = 1
Phase = Literal["my_turn", "opponent_turn"]
Player = Literal["me", "opponent"]
IMAGE_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Turn:
    """One logged turn, with the state before it so it can be undone."""

    n: int
    player: Player
    kind: Literal["play", "pass"]
    word: str = ""
    direction: str = ""  # "across" | "down" | ""
    start: tuple[int, int] = (0, 0)  # zero-based (row, col) of the main word's first letter
    placements: list[dict[str, Any]] = field(default_factory=list)  # row, col, letter, is_blank
    words: list[dict[str, Any]] = field(default_factory=list)  # every word formed: word, points
    points: int = 0
    bingo: int = 0
    source: str = "solver"  # solver | typed | photo | grid
    forced: bool = False  # kept although a word is not in the word list
    at: str = field(default_factory=now_iso)
    before_board: list[str] = field(default_factory=list)
    before_rack: str = ""
    before_base: list[str] = field(default_factory=list)
    before_phase: str = "my_turn"


@dataclass
class Game(Session):
    name: str = ""
    phase: Phase = "my_turn"
    turns: list[Turn] = field(default_factory=list)
    # the board as it was when the pending turn began; new tiles since then are the pending move
    base_board: BoardGrid = field(default_factory=empty_grid)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    ephemeral: bool = False
    image_file: str | None = None
    drop_image: bool = False
    image_sig: int = 0  # identity of the picture last written, so it is only written once

    @property
    def persistent(self) -> bool:
        return not self.ephemeral

    def scores(self) -> dict[str, int]:
        totals = {"me": 0, "opponent": 0}
        for t in self.turns:
            totals[t.player] += t.points
        return totals

    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def clear_picture(self) -> None:
        """The picture no longer matches the board (a move was played since): forget it."""
        self.image = None
        self.recognition = None
        self.drop_image = True

    def follow_board(self) -> None:
        """Before the first logged turn the game has no history: its base is simply the board."""
        if not self.turns and self.phase == "my_turn":
            self.base_board = [row[:] for row in self.board]


class GameStore:
    def __init__(
        self,
        directory: Path | None = None,
        ttl_s: float = 30 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = directory
        self.ttl_s = ttl_s
        self._clock = clock
        self._games: dict[str, Game] = {}
        self._lock = threading.Lock()
        if directory is not None:
            self._load_all()

    # --- access -----------------------------------------------------------------------------

    def create(
        self,
        solver_id: str,
        params: dict[str, Any],
        dictionary_id: str,
        *,
        name: str = "",
        ephemeral: bool = False,
    ) -> Game:
        g = Game(
            id=uuid.uuid4().hex,
            solver_id=solver_id,
            params=dict(params),
            dictionary_id=dictionary_id,
            name=name or f"Game {datetime.now():%b %d, %H:%M}",
            ephemeral=ephemeral,
            last_access=self._clock(),
        )
        with self._lock:
            self._games[g.id] = g
        if not ephemeral:
            self.save(g)
        return g

    def get(self, game_id: str) -> Game:
        """The game, or SessionNotFound. Ephemeral games expire; saved ones never do."""
        now = self._clock()
        with self._lock:
            g = self._games.get(game_id)
            if g is None or (g.ephemeral and now - g.last_access > self.ttl_s):
                self._games.pop(game_id, None)
                raise SessionNotFound(game_id)
            g.last_access = now
        if g.image is None and g.image_file and self.directory is not None:
            path = self.directory / g.image_file
            if path.is_file():
                g.image = path.read_bytes()
                g.image_sig = id(g.image)
        return g

    def find(self, game_id: str) -> Game | None:
        try:
            return self.get(game_id)
        except SessionNotFound:
            return None

    def saved(self) -> list[Game]:
        """Saved games, most recently played first."""
        with self._lock:
            games = [g for g in self._games.values() if g.persistent]
        return sorted(games, key=lambda g: g.updated_at, reverse=True)

    def delete(self, game_id: str) -> bool:
        with self._lock:
            g = self._games.pop(game_id, None)
        if g is None:
            return False
        if self.directory is not None:
            for path in (self.directory / f"{game_id}.json", *self.directory.glob(f"{game_id}.*")):
                path.unlink(missing_ok=True)
        return True

    def sweep(self) -> int:
        now = self._clock()
        with self._lock:
            dead = [
                k
                for k, g in self._games.items()
                if g.ephemeral and now - g.last_access > self.ttl_s
            ]
            for k in dead:
                del self._games[k]
        return len(dead)

    def __len__(self) -> int:
        return len(self._games)

    # --- persistence ------------------------------------------------------------------------

    def save(self, g: Game) -> None:
        g.updated_at = now_iso()
        if g.ephemeral or self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        if g.drop_image:
            if g.image_file:
                (self.directory / g.image_file).unlink(missing_ok=True)
            g.image_file, g.drop_image, g.image_sig = None, False, 0
        if g.image is not None and g.image_sig != id(g.image):
            if g.image_file:
                (self.directory / g.image_file).unlink(missing_ok=True)
            g.image_file = f"{g.id}{IMAGE_EXTENSIONS.get(g.image_type, '.img')}"
            _atomic_write(self.directory / g.image_file, g.image)
            g.image_sig = id(g.image)
        _atomic_write(
            self.directory / f"{g.id}.json",
            json.dumps(game_to_dict(g), indent=1).encode(),
        )

    def _load_all(self) -> None:
        assert self.directory is not None
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.json")):
            try:
                g = game_from_dict(json.loads(path.read_text()), self._clock())
            except Exception:
                log.exception("skipping unreadable game file %s", path.name)
                continue
            self._games[g.id] = g


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# --- (de)serialisation -------------------------------------------------------------------------


def game_to_dict(g: Game) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "id": g.id,
        "name": g.name,
        "created_at": g.created_at,
        "updated_at": g.updated_at,
        "phase": g.phase,
        "solver_id": g.solver_id,
        "params": g.params,
        "dictionary_id": g.dictionary_id,
        "board": grid_to_rows(g.board),
        "rack": g.rack,
        "base_board": grid_to_rows(g.base_board),
        "turns": [asdict(t) for t in g.turns],
        "corrected": g.corrected,
        "recognition": g.recognition.model_dump(mode="json") if g.recognition else None,
        "last_result": g.last_result.model_dump(mode="json") if g.last_result else None,
        "image_file": g.image_file,
        "image_type": g.image_type,
        "fixture_name": g.fixture_name,
    }


def game_from_dict(d: dict[str, Any], clock_now: float) -> Game:
    if d.get("version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported game file version {d.get('version')!r}")
    turns = []
    for t in d.get("turns", []):
        t = dict(t)
        t["start"] = tuple(t.get("start", (0, 0)))
        turns.append(Turn(**t))
    g = Game(
        id=d["id"],
        solver_id=d["solver_id"],
        params=d.get("params", {}),
        dictionary_id=d["dictionary_id"],
        name=d.get("name", ""),
        phase=d.get("phase", "my_turn"),
        turns=turns,
        board=rows_to_grid(d["board"]),
        rack=d.get("rack", ""),
        base_board=rows_to_grid(d.get("base_board") or d["board"]),
        created_at=d.get("created_at", now_iso()),
        updated_at=d.get("updated_at", now_iso()),
        corrected=bool(d.get("corrected", False)),
        image_file=d.get("image_file"),
        image_type=d.get("image_type") or "application/octet-stream",
        fixture_name=d.get("fixture_name"),
        last_access=clock_now,
    )
    if d.get("recognition"):
        g.recognition = RecognitionResultOut.model_validate(d["recognition"])
    if d.get("last_result"):
        g.last_result = SolveResponse.model_validate(d["last_result"])
    return g
