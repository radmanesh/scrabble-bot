"""Games: one saved match, with its turns and every picture taken of it.

A game lives in `<games dir>/<id>.json` with its pictures in `<games dir>/<id>/`. Board rows are
15 strings: "." empty, uppercase a tile, lowercase a blank.

The shape of a game is a locked history plus one open turn:

* `turns` are confirmed and read-only. The first is the starting position (what was on the board
  when tracking began); every later one is a move by you or the opponent.
* The *open turn* is whatever is on `board` that is not on `base_board` (the board as it was when
  the open turn began). It is editable until it is confirmed, whether it came from a suggestion,
  a typed word or a picture.
* `rack` is always editable. `base_rack` is what it was when the open turn began.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from wwf.api.schemas import BOARD_SIZE, BoardGrid, RecognitionResultOut, SolveResponse

log = logging.getLogger("wwf.api")

SCHEMA_VERSION = 2
Phase = Literal["my_turn", "opponent_turn"]
Player = Literal["me", "opponent"]
IMAGE_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def empty_grid() -> BoardGrid:
    return [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]


def grid_to_rows(grid: BoardGrid) -> list[str]:
    return ["".join(c or "." for c in row) for row in grid]


def rows_to_grid(rows: list[str]) -> BoardGrid:
    if len(rows) != BOARD_SIZE or any(len(r) != BOARD_SIZE for r in rows):
        raise ValueError(f"board must be {BOARD_SIZE} rows of {BOARD_SIZE} characters")
    return [[None if ch in ". " else ch for ch in row] for row in rows]


@dataclass
class Turn:
    """One confirmed turn. `kind == "start"` is the position tracking began from."""

    n: int
    player: Player
    kind: Literal["start", "play", "pass"]
    word: str = ""
    direction: str = ""  # "across" | "down" | ""
    start: tuple[int, int] = (0, 0)  # zero-based (row, col) of the main word's first letter
    placements: list[dict[str, Any]] = field(default_factory=list)  # row, col, letter, is_blank
    words: list[dict[str, Any]] = field(default_factory=list)  # every word formed: word, points
    points: int = 0
    bingo: int = 0
    tiles: int = 0  # the starting position: how many tiles it had
    source: str = "typed"  # suggestion | typed | picture | grid | start
    forced: bool = False  # kept although a word is not in the word list
    picture: str | None = None  # the picture this turn was read from, if any
    at: str = field(default_factory=now_iso)
    # what the game looked like before it, so undo can reopen it
    before_base: list[str] = field(default_factory=list)
    before_base_rack: str = ""
    before_phase: str = "my_turn"


@dataclass
class Picture:
    """A picture taken of the game. Once the turn it produced is confirmed, the confirmed board
    is its label: what the reader should have seen."""

    id: str
    file: str
    content_type: str
    source: str  # photo | upload
    turn_n: int  # the turn that was open when it was taken
    taken_at: str = field(default_factory=now_iso)
    read_board: list[str] = field(default_factory=list)  # what the reader said, uncorrected
    read_rack: str = ""
    quad: list[list[float]] | None = None  # where the board was in the picture
    label_board: list[str] | None = None
    label_rack: str | None = None
    label_turn: int | None = None  # the confirmed turn that produced the label


@dataclass
class Game:
    id: str
    solver_id: str
    params: dict[str, Any]
    dictionary_id: str
    name: str = ""
    phase: Phase = "my_turn"  # whose move is next
    started: bool = False  # has the starting position been confirmed?
    board: BoardGrid = field(default_factory=empty_grid)
    base_board: BoardGrid = field(default_factory=empty_grid)  # locked tiles
    rack: str = ""
    base_rack: str = ""
    turns: list[Turn] = field(default_factory=list)
    pictures: list[Picture] = field(default_factory=list)
    open_source: str = ""  # how the open turn's tiles got there; "" when nothing is open
    open_player: Player | None = None  # who the open move is for, if not the player to move
    recognition: RecognitionResultOut | None = None  # the latest reading, for amber flags
    last_result: SolveResponse | None = None  # the latest solve, until the board changes
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    notice: dict[str, Any] | None = field(default=None, repr=False)  # shown once

    # --- derived ----------------------------------------------------------------------------

    def scores(self) -> dict[str, int]:
        totals = {"me": 0, "opponent": 0}
        for t in self.turns:
            totals[t.player] += t.points
        return totals

    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    @property
    def turn_number(self) -> int:
        """The number of the open turn (confirmed turns + 1)."""
        return len(self.turns) + 1

    @property
    def player_to_move(self) -> Player:
        return "me" if self.phase == "my_turn" else "opponent"

    def picture(self, picture_id: str) -> Picture | None:
        return next((p for p in self.pictures if p.id == picture_id), None)

    @property
    def latest_picture(self) -> Picture | None:
        return self.pictures[-1] if self.pictures else None

    def open_pictures(self) -> list[Picture]:
        """Pictures taken during the open turn that have no label yet."""
        return [p for p in self.pictures if p.turn_n == self.turn_number and p.label_board is None]


class GameStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory
        self._games: dict[str, Game] = {}
        self._memory_pictures: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()
        if directory is not None:
            self._load_all()

    # --- access -----------------------------------------------------------------------------

    def create(
        self, solver_id: str, params: dict[str, Any], dictionary_id: str, *, name: str = ""
    ) -> Game:
        g = Game(
            id=uuid.uuid4().hex,
            solver_id=solver_id,
            params=dict(params),
            dictionary_id=dictionary_id,
            name=name or f"Game {datetime.now():%b %d, %H:%M}",
        )
        with self._lock:
            self._games[g.id] = g
        self.save(g)
        return g

    def get(self, game_id: str) -> Game | None:
        with self._lock:
            return self._games.get(game_id)

    def saved(self) -> list[Game]:
        """Games, most recently played first."""
        with self._lock:
            games = list(self._games.values())
        return sorted(games, key=lambda g: g.updated_at, reverse=True)

    def delete(self, game_id: str) -> bool:
        with self._lock:
            g = self._games.pop(game_id, None)
        if g is None:
            return False
        if self.directory is not None:
            (self.directory / f"{game_id}.json").unlink(missing_ok=True)
            shutil.rmtree(self.directory / game_id, ignore_errors=True)
        return True

    def __len__(self) -> int:
        return len(self._games)

    # --- pictures ---------------------------------------------------------------------------

    def picture_dir(self, game_id: str) -> Path | None:
        return None if self.directory is None else self.directory / game_id

    def add_picture(self, g: Game, data: bytes, content_type: str, source: str) -> Picture:
        """Store a picture with the game. In memory-only games the bytes are kept in the store."""
        pid = f"p{len(g.pictures) + 1}"
        ext = IMAGE_EXTENSIONS.get(content_type, ".img")
        pic = Picture(
            id=pid,
            file=f"{pid}{ext}",
            content_type=content_type,
            source=source,
            turn_n=g.turn_number,
        )
        folder = self.picture_dir(g.id)
        if folder is None:
            self._memory_pictures[(g.id, pid)] = data
        else:
            folder.mkdir(parents=True, exist_ok=True)
            _atomic_write(folder / pic.file, data)
        g.pictures.append(pic)
        return pic

    def picture_bytes(self, g: Game, pic: Picture) -> bytes | None:
        folder = self.picture_dir(g.id)
        if folder is None:
            return self._memory_pictures.get((g.id, pic.id))
        path = folder / pic.file
        return path.read_bytes() if path.is_file() else None

    # --- persistence ------------------------------------------------------------------------

    def save(self, g: Game) -> None:
        g.updated_at = now_iso()
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self.directory / f"{g.id}.json", json.dumps(game_to_dict(g), indent=1).encode()
        )

    def _load_all(self) -> None:
        assert self.directory is not None
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text())
                g = game_from_dict(raw)
            except Exception:
                log.exception("skipping unreadable game file %s", path.name)
                continue
            self._games[g.id] = g
            if raw.get("image_file") and not g.pictures:
                self._adopt_legacy_picture(g, raw)

    def _adopt_legacy_picture(self, g: Game, raw: dict[str, Any]) -> None:
        """Games saved before pictures were kept per game had one picture beside the JSON."""
        assert self.directory is not None
        old = self.directory / str(raw["image_file"])
        if not old.is_file():
            return
        content_type = str(raw.get("image_type") or "image/png")
        pic = Picture(
            id="p1",
            file="p1" + IMAGE_EXTENSIONS.get(content_type, old.suffix or ".img"),
            content_type=content_type,
            source="photo",
            turn_n=g.turn_number,
            read_board=grid_to_rows(g.board),
        )
        folder = self.directory / g.id
        folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), folder / pic.file)
        g.pictures.append(pic)
        self.save(g)


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
        "started": g.started,
        "solver_id": g.solver_id,
        "params": g.params,
        "dictionary_id": g.dictionary_id,
        "board": grid_to_rows(g.board),
        "base_board": grid_to_rows(g.base_board),
        "rack": g.rack,
        "base_rack": g.base_rack,
        "open_source": g.open_source,
        "open_player": g.open_player,
        "turns": [asdict(t) for t in g.turns],
        "pictures": [asdict(p) for p in g.pictures],
        "recognition": g.recognition.model_dump(mode="json") if g.recognition else None,
        "last_result": g.last_result.model_dump(mode="json") if g.last_result else None,
    }


def game_from_dict(d: dict[str, Any]) -> Game:
    version = d.get("version")
    if version not in (1, SCHEMA_VERSION):
        raise ValueError(f"unsupported game file version {version!r}")
    turns = []
    for t in d.get("turns", []):
        t = dict(t)
        t["start"] = tuple(t.get("start", (0, 0)))
        # version 1 kept snapshots of the whole board; version 2 only needs the base
        t.pop("before_board", None)
        t.pop("before_rack", None)
        t["before_base_rack"] = t.get("before_base_rack", "")
        turns.append(Turn(**t))
    pictures = [Picture(**p) for p in d.get("pictures", [])]
    g = Game(
        id=d["id"],
        solver_id=d["solver_id"],
        params=d.get("params", {}),
        dictionary_id=d["dictionary_id"],
        name=d.get("name", ""),
        phase=d.get("phase", "my_turn"),
        started=bool(d.get("started", bool(turns))),
        board=rows_to_grid(d["board"]),
        base_board=rows_to_grid(d.get("base_board") or d["board"]),
        rack=d.get("rack", ""),
        base_rack=d.get("base_rack", d.get("rack", "")),
        turns=turns,
        pictures=pictures,
        open_source=d.get("open_source", ""),
        open_player=d.get("open_player"),
        created_at=d.get("created_at", now_iso()),
        updated_at=d.get("updated_at", now_iso()),
    )
    if d.get("recognition"):
        g.recognition = RecognitionResultOut.model_validate(d["recognition"])
    if d.get("last_result"):
        g.last_result = SolveResponse.model_validate(d["last_result"])
    return g
