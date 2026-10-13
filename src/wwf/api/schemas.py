"""Pydantic models shared by all routes.

Boards are 15x15 arrays of null or a one-character string: uppercase for a normal tile, the
lowercase letter for a blank. Racks are strings with "?" for a blank. Coordinates are
[row, col], zero-based.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BOARD_SIZE = 15
MAX_RACK = 7

BoardGrid = list[list[str | None]]


def check_board(board: BoardGrid) -> BoardGrid:
    if len(board) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in board):
        raise ValueError(f"board must be {BOARD_SIZE}x{BOARD_SIZE}")
    out: BoardGrid = []
    for r, row in enumerate(board):
        clean: list[str | None] = []
        for c, cell in enumerate(row):
            if cell is None or cell == "":
                clean.append(None)
            elif len(cell) == 1 and cell.isascii() and cell.isalpha():
                clean.append(cell)
            else:
                raise ValueError(f"cell [{r},{c}] must be null or one letter, got {cell!r}")
        out.append(clean)
    return out


def check_rack(rack: str) -> str:
    rack = "".join(rack.split())
    if len(rack) > MAX_RACK:
        raise ValueError(f"rack holds at most {MAX_RACK} tiles")
    for ch in rack:
        if ch != "?" and not (ch.isascii() and ch.isalpha()):
            raise ValueError(f"rack tile {ch!r} must be a letter or '?'")
    return "".join(ch if ch == "?" else ch.upper() for ch in rack)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Envelope(ApiModel):
    request_id: str = ""


class ErrorBody(ApiModel):
    error: str
    detail: Any = None
    request_id: str = ""


# --- recognition -----------------------------------------------------------------------------

Box = tuple[int, int, int, int]  # x, y, width, height in image pixels


class GridGeometryOut(ApiModel):
    cells: list[list[Box]]
    rack: list[Box]
    pitch: float
    confidence: float
    image_width: int
    image_height: int
    quad: list[list[float]] = []  # board corners: top-left, top-right, bottom-right, bottom-left


class RecognitionResultOut(ApiModel):
    board: BoardGrid
    rack: list[str]
    confidence: list[list[float]]
    needs_review: list[list[bool]]
    geometry: GridGeometryOut | None = None
    warnings: list[str] = []
    input_type: Literal["screenshot", "photo", "unknown"] = "unknown"
    elapsed_ms: int = 0
    # next turn: tiles kept from the previous board, new tiles read, and kept tiles the new
    # picture seems to disagree with (flagged for review, but the previous value is kept)
    kept_from_previous: int = 0
    new_tiles: int = 0
    conflicts: int = 0


class RecognizeResponse(RecognitionResultOut):
    session_id: str
    request_id: str = ""


# --- sessions --------------------------------------------------------------------------------


class BoardIn(ApiModel):
    board: BoardGrid
    rack: str = ""

    _board = field_validator("board")(check_board)
    _rack = field_validator("rack")(check_rack)


class OptionsIn(ApiModel):
    solver_id: str
    params: dict[str, Any] = {}
    dictionary_id: str | None = None


class SolveIn(ApiModel):
    solver_id: str | None = None
    params: dict[str, Any] | None = None
    dictionary_id: str | None = None


class SessionState(Envelope):
    session_id: str
    board: BoardGrid
    rack: str
    has_image: bool
    recognition: RecognitionResultOut | None
    corrected: bool
    solver_id: str
    params: dict[str, Any]
    dictionary_id: str
    has_result: bool
    expires_in_s: int


# --- solving ---------------------------------------------------------------------------------


class PlacementOut(ApiModel):
    row: int
    col: int
    letter: str
    is_blank: bool


class WordOut(ApiModel):
    word: str
    points: int


class ScoreOut(ApiModel):
    total: int
    bingo_bonus: int
    words: list[WordOut]


class MoveOut(ApiModel):
    word: str
    direction: Literal["across", "down"]
    start: tuple[int, int]
    placements: list[PlacementOut]
    score: ScoreOut
    leave: str
    board_after: BoardGrid


class SolveResponse(Envelope):
    solver_id: str
    params: dict[str, Any]
    dictionary_id: str
    elapsed_ms: int
    stats: dict[str, int]
    partial: bool = False
    moves: list[MoveOut]


class SolverParamOut(ApiModel):
    name: str
    label: str
    type: Literal["int", "float", "bool", "choice"]
    default: Any
    min: float | None
    max: float | None
    choices: list[str]
    help: str


class SolverInfoOut(ApiModel):
    id: str
    display_name: str
    description: str
    enabled: bool
    default: bool
    planned: bool
    params: list[SolverParamOut]


class SolversResponse(Envelope):
    default_id: str
    solvers: list[SolverInfoOut]


class DictionaryInfoOut(ApiModel):
    id: str
    label: str
    description: str
    default: bool
    loaded: bool


class DictionariesResponse(Envelope):
    default_id: str
    dictionaries: list[DictionaryInfoOut]


# --- validation and health -------------------------------------------------------------------


class PlacementIn(ApiModel):
    row: int = Field(ge=0, lt=BOARD_SIZE)
    col: int = Field(ge=0, lt=BOARD_SIZE)
    letter: str = Field(min_length=1, max_length=1)
    is_blank: bool | None = None  # default: lowercase letter means blank

    @field_validator("letter")
    @classmethod
    def _letter(cls, v: str) -> str:
        if not (v.isascii() and v.isalpha()):
            raise ValueError("letter must be A-Z")
        return v


class MoveIn(ApiModel):
    placements: list[PlacementIn] = Field(min_length=1, max_length=MAX_RACK)
    direction: Literal["across", "down"] | None = None


class ValidateIn(BoardIn):
    move: MoveIn
    dictionary_id: str | None = None


class ValidateResponse(Envelope):
    violations: list[str]
    score: ScoreOut | None = None  # filled when the move is legal


class HealthResponse(Envelope):
    status: Literal["ok"]
    dictionary_id: str
    dictionary_words: int
    model_loaded: bool
    sessions: int
