"""Move and placement types, plus helpers to build a canonical Move from placements."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from wwf.core.board import Board, Tile


class Direction(Enum):
    ACROSS = "across"
    DOWN = "down"

    @property
    def step(self) -> tuple[int, int]:
        return (0, 1) if self is Direction.ACROSS else (1, 0)

    @property
    def other(self) -> Direction:
        return Direction.DOWN if self is Direction.ACROSS else Direction.ACROSS


@dataclass(frozen=True)
class Placement:
    row: int
    col: int
    tile: Tile


@dataclass(frozen=True)
class Move:
    placements: tuple[Placement, ...]  # only NEW tiles, sorted by (row, col)
    direction: Direction
    main_word: str  # full word including existing tiles
    start: tuple[int, int]  # (row, col) of the main word's first letter

    def key(self) -> frozenset[tuple[int, int, str, bool]]:
        """Identity of the move independent of direction labelling."""
        return frozenset((p.row, p.col, p.tile.letter, p.tile.is_blank) for p in self.placements)

    def transpose(self) -> Move:
        return Move(
            placements=tuple(
                sorted(
                    (Placement(p.col, p.row, p.tile) for p in self.placements),
                    key=lambda p: (p.row, p.col),
                )
            ),
            direction=self.direction.other,
            main_word=self.main_word,
            start=(self.start[1], self.start[0]),
        )


def word_extent(
    board: Board, row: int, col: int, direction: Direction, new: dict[tuple[int, int], Tile]
) -> tuple[tuple[int, int], list[tuple[int, int, Tile]]]:
    """The maximal run of tiles through (row, col) along `direction`, counting `new` tiles.

    Returns the start cell and the list of (row, col, tile) along the run. This sits on the
    solver's scoring hot path, hence the direct indexing.
    """
    cells = board.cells
    size = board.size
    get_new = new.get
    if direction is Direction.ACROSS:
        line = cells[row]
        c = col
        while c > 0 and (line[c - 1] is not None or (row, c - 1) in new):
            c -= 1
        start = (row, c)
        run: list[tuple[int, int, Tile]] = []
        while c < size:
            t = line[c] or get_new((row, c))
            if t is None:
                break
            run.append((row, c, t))
            c += 1
    else:
        r = row
        while r > 0 and (cells[r - 1][col] is not None or (r - 1, col) in new):
            r -= 1
        start = (r, col)
        run = []
        while r < size:
            t = cells[r][col] or get_new((r, col))
            if t is None:
                break
            run.append((r, col, t))
            r += 1
    return start, run


def build_move(
    board: Board, placements: Iterable[Placement], direction: Direction | None = None
) -> Move:
    """Build a canonical Move from new-tile placements.

    Direction is inferred from the placements when they span more than one cell. For a single
    tile, `direction` is honoured if given and it forms a word of 2+ letters that way; otherwise
    ACROSS is preferred when it forms a word, else DOWN. No legality checks happen here.
    """
    ps = tuple(sorted(placements, key=lambda p: (p.row, p.col)))
    if not ps:
        raise ValueError("a move needs at least one placement")
    new = {(p.row, p.col): p.tile for p in ps}
    first = ps[0]
    if len(ps) > 1:
        if all(p.row == first.row for p in ps):
            direction = Direction.ACROSS
        elif all(p.col == first.col for p in ps):
            direction = Direction.DOWN
        # Otherwise the placements are not in line; keep the caller's direction (or ACROSS)
        # so validate_move can report not_in_line.
        direction = direction or Direction.ACROSS
    else:
        candidates = [direction] if direction is not None else []
        candidates += [Direction.ACROSS, Direction.DOWN]
        chosen = None
        for d in candidates:
            _, run = word_extent(board, first.row, first.col, d, new)
            if len(run) >= 2:
                chosen = d
                break
        direction = chosen or (direction or Direction.ACROSS)

    assert direction is not None
    start, run = word_extent(board, first.row, first.col, direction, new)
    word = "".join(t.letter for _, _, t in run)
    return Move(placements=ps, direction=direction, main_word=word, start=start)
