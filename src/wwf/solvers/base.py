"""Solver plugin protocol and shared helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from wwf.core.board import Board
from wwf.core.config import BoardConfig
from wwf.core.dictionary import Dictionary
from wwf.core.move import Move
from wwf.core.rack import BLANK, Rack
from wwf.core.scoring import ScoreBreakdown

ParamType = Literal["int", "float", "bool", "choice"]


@dataclass(frozen=True)
class SolverParam:
    name: str
    label: str
    type: ParamType
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""


@dataclass(frozen=True)
class ScoredMove:
    move: Move
    score: ScoreBreakdown
    leave: str  # tiles left on rack


@dataclass(frozen=True)
class SolverResult:
    moves: tuple[ScoredMove, ...]  # sorted by score desc, then word asc
    elapsed_ms: int
    stats: dict[str, int]


@runtime_checkable
class Solver(Protocol):
    id: str
    display_name: str
    description: str
    params: tuple[SolverParam, ...]

    def solve(
        self, board: Board, rack: Rack, dictionary: Dictionary, cfg: BoardConfig, **params: Any
    ) -> SolverResult: ...


class ParamError(ValueError):
    """A solver parameter is unknown, of the wrong type, or out of range."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def _coerce(p: SolverParam, value: Any) -> Any:
    if p.type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "on", "off"):
            return value.lower() in ("true", "1", "on")
        raise ValueError("expected a boolean")
    if p.type == "int":
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError("expected an integer")
        out: int | float = int(value)
    elif p.type == "float":
        if isinstance(value, bool):
            raise ValueError("expected a number")
        out = float(value)
    else:
        if value not in p.choices:
            raise ValueError(f"expected one of {list(p.choices)}")
        return value
    if p.min is not None and out < p.min:
        raise ValueError(f"must be >= {p.min:g}")
    if p.max is not None and out > p.max:
        raise ValueError(f"must be <= {p.max:g}")
    return out


def resolve_params(declared: Iterable[SolverParam], given: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults, coerce types, check ranges. Raises ParamError listing every problem."""
    decl = {p.name: p for p in declared}
    errors = [f"{name}: unknown parameter" for name in given if name not in decl]
    out: dict[str, Any] = {}
    for name, p in decl.items():
        if name not in given or given[name] is None:
            out[name] = p.default
            continue
        try:
            out[name] = _coerce(p, given[name])
        except (TypeError, ValueError) as e:
            errors.append(f"{name}: {e}")
    if errors:
        raise ParamError(errors)
    return out


def leave_after(rack: Rack, move: Move) -> str:
    """Rack tiles left after playing `move`, in rack order."""
    used = Counter(BLANK if p.tile.is_blank else p.tile.letter for p in move.placements)
    left: list[str] = []
    for t in rack.tiles:
        if used[t] > 0:
            used[t] -= 1
        else:
            left.append(t)
    return "".join(left)


VOWELS = frozenset("AEIOU")


def leave_heuristic(leave: str) -> float:
    """Crude leave quality (higher is better): balanced vowels/consonants, no Q without U,
    few duplicates, blanks and S are worth keeping. Not a real leave table."""
    if not leave:
        return 0.0
    letters = [t for t in leave if t != BLANK]
    vowels = sum(t in VOWELS for t in letters)
    consonants = len(letters) - vowels
    value = -1.5 * abs(vowels - consonants * 0.7)
    if "Q" in letters and "U" not in letters:
        value -= 8.0
    counts = Counter(letters)
    value -= sum(n - 1 for n in counts.values() if n > 1)
    value += 3.0 * leave.count(BLANK) + 1.5 * counts["S"]
    return value


def sort_key_score_only(m: ScoredMove) -> tuple[Any, ...]:
    return (-m.score.total, m.move.main_word, m.move.start, m.move.direction.value)


def sort_key_score_then_leave(m: ScoredMove) -> tuple[Any, ...]:
    return (
        -m.score.total,
        -leave_heuristic(m.leave),
        m.move.main_word,
        m.move.start,
        m.move.direction.value,
    )
